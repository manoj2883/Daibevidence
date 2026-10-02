"""
v3 retrieval: dense + BM25 over chunks, fused, expanded to full parent documents.

1. Dense: top V3_DENSE_TOP_K chunks from Pinecone for the rewritten search_query,
   across both tiers (PubMed chunks and background pages) on every question.
2. BM25: top V3_BM25_TOP_K chunks from a local rank_bm25 index over the same
   chunks, scored on the query step's keywords.
3. Reciprocal rank fusion: score = sum over lists of 1 / (V3_RRF_K + rank).
4. Dedupe by parent (PMID or background page), keeping each parent's best chunk.
5. Keep the top V3_FINAL_PARENTS parents and swap each matched chunk for its FULL
   parent text, which is what the evidence judge and generation see.
6. Order: strongest first, second-strongest last, weaker ones in the middle
   (models attend most to the start and end of a long context).

The chunk corpus and parents are read from corpus/v3_5k_recursive.json.gz, the
same file the ingestion script uploaded from, so BM25 and Pinecone index the
exact same chunks.
"""
import gzip
import json
import re
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.ingest.config import (
    NAMESPACE_V3,
    V3_BM25_TOP_K,
    V3_CORPUS_PATH,
    V3_DENSE_TOP_K,
    V3_FINAL_PARENTS,
    V3_RRF_K,
)
from src.rag.retriever import retrieve
from src.rag.types import RetrievedChunk

_STOPWORDS = set(
    "a an and are as at be by for from has have how in is it its of on or that the this to was were what when "
    "which who why will with does do did can should i my me you your we our there their than then into about".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")

_lock = threading.Lock()
_corpus: Optional[Dict[str, Any]] = None


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOPWORDS]


def load_corpus(path: str = V3_CORPUS_PATH) -> Dict[str, Any]:
    """Loads chunks + parents and builds the BM25 index once per process."""
    global _corpus
    if _corpus is not None:
        return _corpus
    with _lock:
        if _corpus is None:
            from rank_bm25 import BM25Okapi

            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
            chunks = data["chunks"]
            data["bm25"] = BM25Okapi([tokenize(c["text"]) for c in chunks])
            data["by_id"] = {c["id"]: c for c in chunks}
            _corpus = data
    return _corpus


def bm25_search(keywords: Sequence[str], fallback_query: str, k: int = V3_BM25_TOP_K) -> List[Tuple[str, float]]:
    corpus = load_corpus()
    terms = tokenize(" ".join(keywords)) if keywords else tokenize(fallback_query)
    if not terms:
        return []
    scores = corpus["bm25"].get_scores(terms)
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(corpus["chunks"][i]["id"], float(scores[i])) for i in top if scores[i] > 0]


def dense_search(search_query: str, namespace: str, k: int = V3_DENSE_TOP_K) -> List[Tuple[str, float, Dict[str, Any]]]:
    hits = retrieve(search_query, k=k, namespace=namespace)
    out = []
    for h in hits:
        meta = h.metadata or {}
        chunk_id = f"{meta.get('parent_id')}_{meta.get('chunk_index')}"
        out.append((chunk_id, h.score, {**meta, "text": h.text}))
    return out


def reciprocal_rank_fusion(ranked_lists: Sequence[Sequence[str]], k: int = V3_RRF_K) -> Dict[str, float]:
    fused: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused


def dedupe_by_parent(fused: Dict[str, float], parent_of: Dict[str, str], limit: int) -> List[Tuple[str, str, float]]:
    """[(parent_id, best_chunk_id, fused_score)], best first, one entry per parent."""
    best: Dict[str, Tuple[str, float]] = {}
    for chunk_id, score in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0])):
        parent = parent_of[chunk_id]
        if parent not in best:
            best[parent] = (chunk_id, score)
    ordered = sorted(best.items(), key=lambda kv: (-kv[1][1], kv[0]))
    return [(p, c, s) for p, (c, s) in ordered[:limit]]


def order_passages(items: List[Any]) -> List[Any]:
    """Strongest first, second-strongest last, the rest (in rank order) in the middle."""
    if len(items) < 3:
        return list(items)
    return [items[0]] + list(items[2:]) + [items[1]]


def hybrid_retrieve(
    search_query: str,
    keywords: Sequence[str],
    namespace: str = NAMESPACE_V3,
    final_k: int = V3_FINAL_PARENTS,
) -> Tuple[List[RetrievedChunk], Dict[str, Any]]:
    """
    Returns (parent passages in prompt order, inspector details). Each passage's
    text is the full parent document; metadata carries the matched chunk too.
    """
    corpus = load_corpus()
    dense = dense_search(search_query, namespace)
    bm25 = bm25_search(keywords, search_query)

    parent_of = {c["id"]: c["metadata"]["parent_id"] for c in corpus["chunks"]}
    for chunk_id, _, meta in dense:
        parent_of.setdefault(chunk_id, meta.get("parent_id"))

    fused = reciprocal_rank_fusion([[c for c, _, _ in dense], [c for c, _ in bm25]])
    top = dedupe_by_parent(fused, parent_of, final_k)

    dense_rank = {c: i for i, (c, _, _) in enumerate(dense, start=1)}
    dense_score = {c: s for c, s, _ in dense}
    bm25_rank = {c: i for i, (c, _) in enumerate(bm25, start=1)}

    passages = []
    for rank, (parent_id, chunk_id, score) in enumerate(top, start=1):
        parent = corpus["parents"][parent_id]
        chunk = corpus["by_id"].get(chunk_id, {})
        metadata = {k: v for k, v in parent.items() if k != "text"}
        metadata.update({
            "parent_id": parent_id,
            "matched_chunk_id": chunk_id,
            "matched_chunk_text": chunk.get("text", ""),
            "fusion_rank": rank,
            "dense_rank": dense_rank.get(chunk_id),
            "dense_score": dense_score.get(chunk_id),
            "bm25_rank": bm25_rank.get(chunk_id),
        })
        passages.append(RetrievedChunk(text=parent["text"], metadata=metadata, score=score))

    inspector = {
        "namespace": namespace,
        "dense": [{"chunk_id": c, "score": round(s, 4)} for c, s, _ in dense],
        "bm25": [{"chunk_id": c, "score": round(s, 3)} for c, s in bm25],
        "fused": [{"parent_id": p, "chunk_id": c, "score": round(s, 5)} for p, c, s in top],
    }
    return order_passages(passages), inspector


def dense_chunks_retrieve(question: str, namespace: str, k: int = V3_FINAL_PARENTS) -> Tuple[List[RetrievedChunk], Dict[str, Any]]:
    """
    Eval Run A: plain dense top-k chunks on the original question, both tiers, no
    rewriting, no BM25, no parent expansion, no reordering (v2_5k + background).
    """
    hits = retrieve(question, k=k, namespace=namespace)
    inspector = {
        "namespace": namespace,
        "dense": [{"chunk_id": f"{(h.metadata or {}).get('pmid') or (h.metadata or {}).get('title')}_{(h.metadata or {}).get('chunk_index')}",
                   "score": round(h.score, 4)} for h in hits],
        "bm25": [],
        "fused": [],
    }
    return hits, inspector
