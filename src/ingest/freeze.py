"""
Corpus freeze manifest: a single file documenting exactly what was pulled,
when, and how, so the corpus behind a given set of evaluation results is
reproducible and auditable later — PubMed's own result set for the same
query shifts over time as new articles are indexed, so "re-run the same
query" is not a valid way to reconstruct a prior corpus.
"""
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.ingest.config import CORPUS_MANIFEST_PATH, FETCH_META_PATH
from src.ingest.local_embeddings import EMBEDDING_DIM, MODEL_NAME


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _load_fetch_meta(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_corpus_manifest(
    documents: List[Dict],
    chunks: List[Dict],
    uploaded_to_pinecone: bool,
    pinecone_index_name: Optional[str] = None,
    fetch_meta_path: str = FETCH_META_PATH,
    out_path: str = CORPUS_MANIFEST_PATH,
) -> Dict:
    """
    Write a manifest recording the fetch date/query provenance (from
    pubmed.py's fetch-meta file, if one exists) plus corpus-level stats
    (population/topic/publication-type breakdowns, chunk count, embedding
    model, git commit). Call this once you're happy with a corpus and want
    to freeze/record it — re-running ingestion overwrites this file, so if
    you want to keep an old manifest as a permanent record, copy it out
    (or commit it to git; data/corpus_manifest.json is exempted from
    .gitignore for exactly this reason) before re-running.
    """
    fetch_meta = _load_fetch_meta(fetch_meta_path)

    population_counts = Counter(doc.get("population", "") for doc in documents)
    topic_counts = Counter(doc.get("topic", "") for doc in documents)
    publication_type_counts = Counter()
    for doc in documents:
        for pt in (doc.get("publication_type") or "").split(";"):
            pt = pt.strip()
            if pt:
                publication_type_counts[pt] += 1

    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "fetch": fetch_meta or {"note": "No fetch-meta file found — corpus may have been loaded via --skip-fetch from an untracked cache."},
        "corpus_stats": {
            "num_abstracts": len(documents),
            "num_chunks": len(chunks),
            "population_counts": dict(population_counts),
            "topic_counts": dict(topic_counts),
            "publication_type_counts": dict(publication_type_counts),
        },
        "embedding_model": MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "pinecone_index": pinecone_index_name,
        "uploaded_to_pinecone": uploaded_to_pinecone,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest
