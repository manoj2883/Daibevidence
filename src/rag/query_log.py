"""
Append-only query log: every question, what was retrieved for it, and the
final answer. This is the raw data an evaluation section needs (retrieval
hit rate, citation accuracy, refusal/mismatch behavior) — it can't be
reconstructed after the fact, so every /query call is logged, not just the
ones used for a formal eval run.
"""
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.ingest.config import QUERY_LOG_PATH
from src.rag.types import RetrievedChunk


def _chunk_summary(chunk: RetrievedChunk) -> Dict[str, Any]:
    meta = chunk.metadata or {}
    return {
        "pmid": meta.get("pmid", ""),
        "title": meta.get("title", ""),
        "chunk_index": meta.get("chunk_index", ""),
        "population": meta.get("population", ""),
        "publication_type": meta.get("publication_type", ""),
        "score": float(chunk.score),
    }


def log_query_event(
    question: str,
    requested_population: str,
    chunks: List[RetrievedChunk],
    answer: str,
    path: str = QUERY_LOG_PATH,
    state: Optional[str] = None,
) -> None:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "requested_population": requested_population,
        "state": state,
        "retrieved_chunks": [_chunk_summary(chunk) for chunk in chunks],
        "answer": answer,
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
