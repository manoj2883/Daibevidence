import os
from typing import List, Optional

from dotenv import load_dotenv
from pinecone import Pinecone

from src.ingest.config import (
    LOW_CONFIDENCE_MARGIN,
    NAMESPACE_V2_5K,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_TOP_K,
    SIMILARITY_FLOOR_DEFAULT,
)
from src.ingest.local_embeddings import LocalSentenceTransformerEmbeddings
from src.ingest.uploader import TEXT_METADATA_KEY
from src.rag.types import RetrievalDecision, RetrievedChunk

# Load environment variables
load_dotenv()

_index_cache = {}
_embeddings_cache: Optional[LocalSentenceTransformerEmbeddings] = None


def get_index():
    """
    Initialize and return the raw Pinecone index handle.
    """
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY must be set in environment variables.")
    pinecone_key = pinecone_key.strip()

    index_name = os.environ.get("PINECONE_INDEX_NAME")
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME must be set in environment variables.")
    index_name = index_name.strip()

    cache_key = (pinecone_key, index_name)
    if cache_key not in _index_cache:
        pc = Pinecone(api_key=pinecone_key)
        _index_cache[cache_key] = pc.Index(index_name)
    return _index_cache[cache_key]


def get_retriever_k() -> int:
    return int(os.environ.get("RETRIEVER_TOP_K", str(RETRIEVAL_TOP_K)))


def get_candidate_k() -> int:
    return int(os.environ.get("RETRIEVAL_CANDIDATE_K", str(RETRIEVAL_CANDIDATE_K)))


def get_similarity_floor() -> float:
    return float(os.environ.get("SIMILARITY_FLOOR", str(SIMILARITY_FLOOR_DEFAULT)))


def get_similarity_floor_background() -> Optional[float]:
    """
    Phase 4: the background tier can be tuned independently, since
    background (patient-education) and evidence (PubMed) text embed with
    different score distributions. Returns None (meaning "use the same
    floor as evidence") unless SIMILARITY_FLOOR_BACKGROUND is explicitly
    set, so nothing changes for existing deployments until it's tuned.
    """
    value = os.environ.get("SIMILARITY_FLOOR_BACKGROUND")
    return float(value) if value is not None else None


def get_namespace() -> str:
    """
    Which Pinecone namespace production queries hit. Defaults to
    NAMESPACE_V2_5K ("v2_5k") — the live corpus — rather than "" (the
    index's default namespace), which used to be the silent fallback and
    is exactly the bug this default closes: querying "" looks identical to
    a successful query, it just silently returns nothing meaningful from
    the wrong (empty, in this index) namespace. Eval scripts and
    materialize_v1_300_namespace.py override this per-call via retrieve()'s
    own `namespace` argument (e.g. --namespace v1_300), which always wins
    over this default — see resolve_namespace().
    """
    return os.environ.get("PINECONE_NAMESPACE", NAMESPACE_V2_5K)


def resolve_namespace(namespace: Optional[str] = None) -> str:
    """
    The single place that turns an optional per-call namespace override
    into the namespace that will actually be queried: the override if one
    was passed, otherwise get_namespace()'s default. Shared by retrieve()
    and src.rag.chain.stream_answer so the UI's retrieval inspector always
    reports the namespace that was actually queried, not a guess.
    """
    return get_namespace() if namespace is None else namespace


def get_namespace_stats(namespace: str) -> Optional[dict]:
    """
    The namespace's stats sub-dict from Pinecone's describe_index_stats()
    (currently just {"vector_count": N}), or None if the namespace doesn't
    exist in the index at all. Querying a nonexistent or empty namespace
    with index.query() doesn't error — it just returns zero matches, which
    is indistinguishable from "this question has no relevant evidence".
    describe_index_stats() is the only way to tell those apart before
    ever sending a real query.
    """
    index = get_index()
    stats = index.describe_index_stats()
    namespaces = stats.get("namespaces") or {}
    return namespaces.get(namespace)


def validate_namespace(namespace: str) -> int:
    """
    Fail-fast startup check: refuse to start rather than silently querying
    an empty or nonexistent namespace (see get_namespace_stats). Returns
    the vector count on success; raises RuntimeError naming both the
    namespace and the index on failure — never falls back to another
    namespace, since a silent fallback is exactly the bug being fixed.
    """
    index_name = os.environ.get("PINECONE_INDEX_NAME", "").strip()
    stats = get_namespace_stats(namespace)
    vector_count = (stats or {}).get("vector_count", 0)
    if not stats or vector_count == 0:
        raise RuntimeError(
            f"Pinecone namespace {namespace!r} does not exist or has zero vectors in "
            f"index {index_name!r}. Refusing to start rather than silently falling back "
            f"to another namespace. Set the PINECONE_NAMESPACE env var to a populated "
            f"namespace, or ingest data into {namespace!r} first "
            f"(e.g. scripts/fetch_corpus.py or scripts/materialize_v1_300_namespace.py)."
        )
    return vector_count


def retrieve(question: str, k: Optional[int] = None, namespace: Optional[str] = None) -> List[RetrievedChunk]:
    """
    Embed the question locally and query Pinecone for the top-k most similar
    chunks. Returns them ordered by descending similarity score, exactly as
    Pinecone returns them — never more than k.
    """
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = LocalSentenceTransformerEmbeddings()

    k = k or get_retriever_k()
    namespace = resolve_namespace(namespace)
    query_vector = _embeddings_cache.embed_query(question)

    index = get_index()
    response = index.query(vector=query_vector, top_k=k, include_metadata=True, namespace=namespace)

    chunks = []
    for match in response.get("matches", []):
        metadata = dict(match.get("metadata") or {})
        text = metadata.pop(TEXT_METADATA_KEY, "")
        chunks.append(RetrievedChunk(text=text, metadata=metadata, score=float(match.get("score", 0.0))))
    return chunks


def decide_retrieval_state(
    candidates: List[RetrievedChunk],
    floor: float,
    low_confidence_margin: float = LOW_CONFIDENCE_MARGIN,
    background_floor: Optional[float] = None,
) -> RetrievalDecision:
    """
    Pure decision logic over an already-retrieved candidate pool — no I/O,
    so it's directly unit-testable and reusable by scripts/tune_threshold.py
    without re-querying Pinecone per floor value.

    Candidates are assumed sorted descending by score (as Pinecone returns
    them). A chunk survives if its score clears its tier's floor: `floor`
    for evidence-tier chunks, `background_floor` for background-tier ones
    (Phase 4 — background and evidence text embed with different score
    distributions, so they can be tuned independently). `background_floor`
    defaults to `floor` itself, preserving the original single-floor
    behavior when not given. If nothing survives in either tier,
    "out_of_scope" — no Claude call is made at all. If the best surviving
    score is within low_confidence_margin of the evidence floor,
    "answered_low_confidence". Otherwise "answered". Note: "answered"/
    "answered_low_confidence" here are a provisional, pre-generation
    signal across both tiers combined — src.rag.chain determines the
    authoritative final state (which can become "no_evidence_for_claim")
    after seeing which chunks the model actually cites.
    """
    effective_background_floor = floor if background_floor is None else background_floor

    def _floor_for(chunk: RetrievedChunk) -> float:
        is_background = (chunk.metadata or {}).get("source_type") == "background"
        return effective_background_floor if is_background else floor

    candidate_scores = [c.score for c in candidates]
    surviving = [c for c in candidates if c.score >= _floor_for(c)]

    if not surviving:
        state = "out_of_scope"
    elif surviving[0].score < floor + low_confidence_margin:
        state = "answered_low_confidence"
    else:
        state = "answered"

    return RetrievalDecision(
        state=state,
        surviving_chunks=surviving,
        candidate_scores=candidate_scores,
        floor=floor,
        background_floor=background_floor,
        low_confidence_margin=low_confidence_margin,
    )


def retrieve_with_floor(question: str, namespace: Optional[str] = None) -> RetrievalDecision:
    """
    The Task 1 retrieval path: pull a wide candidate pool and let the
    similarity floor decide how many (if any) are relevant enough to use,
    instead of always forcing a fixed top-k. `namespace` lets Phase 4's
    evaluation run the same question set against v1_300 and v2_5k
    separately for a side-by-side comparison, without touching production
    (which stays on get_namespace()'s default unless overridden).
    """
    candidates = retrieve(question, k=get_candidate_k(), namespace=namespace)
    return decide_retrieval_state(
        candidates, get_similarity_floor(), background_floor=get_similarity_floor_background()
    )
