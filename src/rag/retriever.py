import os
from typing import List, Optional

from dotenv import load_dotenv
from pinecone import Pinecone

from src.ingest.config import (
    LOW_CONFIDENCE_MARGIN,
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


def retrieve(question: str, k: Optional[int] = None) -> List[RetrievedChunk]:
    """
    Embed the question locally and query Pinecone for the top-k most similar
    chunks. Returns them ordered by descending similarity score, exactly as
    Pinecone returns them — never more than k.
    """
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = LocalSentenceTransformerEmbeddings()

    k = k or get_retriever_k()
    query_vector = _embeddings_cache.embed_query(question)

    index = get_index()
    response = index.query(vector=query_vector, top_k=k, include_metadata=True)

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
) -> RetrievalDecision:
    """
    Pure decision logic over an already-retrieved candidate pool — no I/O,
    so it's directly unit-testable and reusable by scripts/tune_threshold.py
    without re-querying Pinecone per floor value.

    Candidates are assumed sorted descending by score (as Pinecone returns
    them). A chunk survives if its score >= floor. If nothing survives,
    "out_of_scope" — nothing in either tier is relevant enough to answer
    from, so no Claude call is made at all. If the best surviving score is
    within low_confidence_margin of the floor, "answered_low_confidence".
    Otherwise "answered". Note: "answered"/"answered_low_confidence" here
    are a provisional, pre-generation signal across both tiers combined —
    src.rag.chain determines the authoritative final state (which can
    become "no_evidence_for_claim") after seeing which chunks the model
    actually cites.
    """
    candidate_scores = [c.score for c in candidates]
    surviving = [c for c in candidates if c.score >= floor]

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
        low_confidence_margin=low_confidence_margin,
    )


def retrieve_with_floor(question: str) -> RetrievalDecision:
    """
    The Task 1 retrieval path: pull a wide candidate pool and let the
    similarity floor decide how many (if any) are relevant enough to use,
    instead of always forcing a fixed top-k.
    """
    candidates = retrieve(question, k=get_candidate_k())
    return decide_retrieval_state(candidates, get_similarity_floor())
