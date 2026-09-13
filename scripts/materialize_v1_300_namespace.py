"""
Phase 3: copy the existing 300-abstract evidence corpus + background tier
into a literal Pinecone namespace ("v1_300"), re-using the already-cached
chunk JSON and the exact same stable IDs originally used to upload them.

This is a copy, not a move — the original vectors in the default
namespace are never touched, read, or deleted by this script. Re-running
it is safe (same IDs => idempotent upsert into v1_300).

Rationale: v1_300 needs to be a real, queryable Pinecone namespace (not
just "the default namespace, relabeled") so scripts/run_evaluation.py can
run the identical question set against v1_300 and v2_5k side-by-side with
one code path (retrieve_with_floor(question, namespace=...)), and so the
default namespace can remain the untouched, never-modified baseline the
brief requires ("Never delete or overwrite the original 300-abstract
corpus").

Usage:
    python -m scripts.materialize_v1_300_namespace
    python -m scripts.materialize_v1_300_namespace --dry-run
"""
import argparse
import json
import os

from dotenv import load_dotenv

from src.ingest.config import (
    BACKGROUND_CHUNKS_JSON_PATH,
    CHUNKS_JSON_PATH,
    FETCH_META_PATH,
    NAMESPACE_V1_300,
    RAW_JSON_PATH,
    V1_300_MANIFEST_PATH,
)
from src.ingest.freeze import write_corpus_manifest
from src.ingest.run_background_ingest import _stable_id as _background_stable_id
from src.ingest.uploader import upload_chunks


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Copy the existing v1_300 corpus into its own Pinecone namespace.")
    parser.add_argument("--namespace", default=NAMESPACE_V1_300)
    parser.add_argument("--dry-run", action="store_true", help="Load/validate the caches but skip the Pinecone upload.")
    args = parser.parse_args()

    for path in (CHUNKS_JSON_PATH, BACKGROUND_CHUNKS_JSON_PATH, RAW_JSON_PATH):
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found — this script expects the original v1_300 caches to already exist.")

    with open(CHUNKS_JSON_PATH, encoding="utf-8") as f:
        evidence_chunks = json.load(f)
    with open(BACKGROUND_CHUNKS_JSON_PATH, encoding="utf-8") as f:
        background_chunks = json.load(f)
    with open(RAW_JSON_PATH, encoding="utf-8") as f:
        raw_documents = json.load(f)

    print(f"Loaded {len(evidence_chunks)} evidence chunks and {len(background_chunks)} background chunks from cache.")

    evidence_ids = [f"{c['metadata']['pmid']}_{c['metadata']['chunk_index']}" for c in evidence_chunks]
    background_ids = [
        _background_stable_id(c["metadata"]["publisher"], c["metadata"]["title"], c["metadata"]["chunk_index"])
        for c in background_chunks
    ]

    if args.dry_run:
        print(f"Dry run — would upsert {len(evidence_chunks) + len(background_chunks)} chunks into namespace {args.namespace!r}.")
        return

    print(f"Embedding and upserting {len(evidence_chunks)} evidence chunks into namespace {args.namespace!r}...")
    upload_chunks(evidence_chunks, ids=evidence_ids, namespace=args.namespace)
    print(f"Embedding and upserting {len(background_chunks)} background chunks into namespace {args.namespace!r}...")
    upload_chunks(background_chunks, ids=background_ids, namespace=args.namespace)
    print("v1_300 namespace materialized. Default namespace (the original vectors) was never read or modified.")

    manifest = write_corpus_manifest(
        raw_documents,
        evidence_chunks + background_chunks,
        uploaded_to_pinecone=True,
        pinecone_index_name=None,
        pinecone_namespace=args.namespace,
        fetch_meta_path=FETCH_META_PATH,
        out_path=V1_300_MANIFEST_PATH,
    )
    print(f"Wrote corpus manifest to {V1_300_MANIFEST_PATH}.")
    print(json.dumps(manifest["corpus_stats"], indent=2))


if __name__ == "__main__":
    main()
