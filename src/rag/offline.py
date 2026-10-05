"""
OFFLINE_MODE=1: answer with the most relevant studies, no Anthropic or Pinecone calls.

Searches the chunks saved in corpus/v3_5k_recursive.json.gz entirely in memory:
- dense: all-MiniLM-L6-v2 (local fastembed model) over the chunk texts,
- BM25: the same in-memory index the online pipeline uses (src.rag.hybrid_retriever),
- reciprocal rank fusion, one entry per paper, top OFFLINE_TOP_K papers.
The query rewrite, evidence judge, generation and verifier all need Anthropic, so they are skipped:
the user gets the studies, not a written answer.

The chunk vectors aren't in the corpus file, so the first offline start embeds all chunks
(a few minutes on CPU) and caches them in data/offline_index_v3.npz. Later starts load the cache.
The cache is rebuilt automatically if the corpus file changes. Pre-build it with
`python -m scripts.build_offline_index`.
"""
import hashlib
import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from src.ingest.local_embeddings import LocalSentenceTransformerEmbeddings
from src.rag.hybrid_retriever import bm25_search, dedupe_by_parent, load_corpus, reciprocal_rank_fusion

OFFLINE_TEXT = "Offline mode: showing the most relevant studies"
OFFLINE_TOP_K = 5
OFFLINE_CANDIDATES = 20
SNIPPET_CHARS = 320
OFFLINE_INDEX_PATH = "data/offline_index_v3.npz"

_lock = threading.Lock()
_index: Optional[Dict[str, Any]] = None


def offline_mode() -> bool:
    return os.environ.get("OFFLINE_MODE", "").strip().lower() in ("1", "true", "yes")


def _corpus_fingerprint(chunk_ids: List[str]) -> str:
    return hashlib.sha256("\n".join(chunk_ids).encode("utf-8")).hexdigest()[:16]


def build_offline_index(path: str = OFFLINE_INDEX_PATH, batch: int = 512, verbose: bool = True) -> Dict[str, Any]:
    """Embed every corpus chunk with the local model and save the unit-normalized vectors (float16)."""
    corpus = load_corpus()
    chunks = corpus["chunks"]
    ids = [c["id"] for c in chunks]
    model = LocalSentenceTransformerEmbeddings()
    parts = []
    for start in range(0, len(chunks), batch):
        parts.append(np.asarray(model.embed_documents([c["text"] for c in chunks[start:start + batch]]), dtype=np.float32))
        if verbose:
            print(f"  embedded {min(start + batch, len(chunks))}/{len(chunks)} chunks", flush=True)
    vectors = np.vstack(parts)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, vectors=vectors.astype(np.float16), fingerprint=_corpus_fingerprint(ids))
    return {"vectors": vectors, "ids": ids}


def load_offline_index(path: str = OFFLINE_INDEX_PATH) -> Dict[str, Any]:
    """The cached chunk vectors, built on first use (or when the corpus changed)."""
    global _index
    if _index is not None:
        return _index
    with _lock:
        if _index is None:
            ids = [c["id"] for c in load_corpus()["chunks"]]
            cached = None
            if os.path.exists(path):
                data = np.load(path)
                if str(data["fingerprint"]) == _corpus_fingerprint(ids) and len(data["vectors"]) == len(ids):
                    cached = {"vectors": data["vectors"].astype(np.float32), "ids": ids}
            if cached is None:
                print(f"[offline] building {path} (one-time, a few minutes)...", flush=True)
                cached = build_offline_index(path)
            _index = cached
    return _index


def offline_search(question: str, k: int = OFFLINE_TOP_K) -> List[Dict[str, Any]]:
    """Top-k papers for a question: local dense + BM25, fused by rank, one entry per paper."""
    corpus = load_corpus()
    index = load_offline_index()
    query = np.asarray(LocalSentenceTransformerEmbeddings().embed_query(question), dtype=np.float32)
    query /= max(float(np.linalg.norm(query)), 1e-12)
    scores = index["vectors"] @ query
    dense = [index["ids"][i] for i in np.argsort(-scores)[:OFFLINE_CANDIDATES]]
    bm25 = [chunk_id for chunk_id, _ in bm25_search([], question, k=OFFLINE_CANDIDATES)]

    parent_of = {c["id"]: c["metadata"]["parent_id"] for c in corpus["chunks"]}
    fused = reciprocal_rank_fusion([dense, bm25])
    results = []
    for rank, (parent_id, chunk_id, score) in enumerate(dedupe_by_parent(fused, parent_of, k), start=1):
        parent = corpus["parents"][parent_id]
        text = parent.get("text", "")
        results.append({
            "rank": rank,
            "source_type": parent.get("source_type", "evidence"),
            "pmid": parent.get("pmid", ""),
            "title": parent.get("title", ""),
            "year": parent.get("year", ""),
            "journal": parent.get("journal", ""),
            "publisher": parent.get("publisher", ""),
            "source_url": parent.get("source_url", ""),
            "population": parent.get("population", ""),
            "evidence_badge": parent.get("study_design", ""),
            "snippet": text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…",
            "found_by": [name for name, hits in (("dense", dense), ("bm25", bm25)) if chunk_id in hits],
        })
    return results


def offline_events(question: str):
    """Event stream for OFFLINE_MODE: one "offline" event with the top studies, then "done"."""
    sources = offline_search(question)
    yield {"event": "offline", "data": {"status": "offline", "state": "offline", "message": OFFLINE_TEXT, "sources": sources}}
    yield {"event": "done", "data": {"status": "offline", "state": "offline"}}
