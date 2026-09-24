"""
Migration: tag every existing study/evidence-tier vector with
source_type="evidence", in a given namespace.

Needed because the background tier introduces source_type as a
distinguishing field (evidence vs background), but every PubMed-derived
chunk ingested before src.ingest.chunker started setting it explicitly
predates the field entirely. A missing metadata field doesn't satisfy a
Pinecone filter — not even via $ne — so those chunks are simply invisible
to a tier-filtered query (`retrieve(..., tier="study")`) until this runs;
that's not a hypothetical, it's how the initial tier-filter query against
v2_5k returned zero matches while developing this script.

Re-embeds and re-upserts (via upload_chunks, same stable IDs — safe to
re-run) rather than a per-ID metadata-only PATCH loop: identical text
under the same embedding model produces identical vector values, so this
is a same-namespace no-op except for the added source_type field, and it's
far fewer, larger batched requests than one PATCH call per chunk.

Usage:
    python -m scripts.backfill_source_type --namespace v2_5k --chunks-json data/v2_5k_chunks.json --dry-run
    python -m scripts.backfill_source_type --namespace v2_5k --chunks-json data/v2_5k_chunks.json
    python -m scripts.backfill_source_type --namespace v1_300 --chunks-json data/pubmed_chunks.json
"""
import argparse
import json

from dotenv import load_dotenv

from src.ingest.config import CHUNKS_JSON_PATH
from src.ingest.uploader import upload_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill source_type=evidence onto existing Pinecone vectors in a namespace.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would change, touch nothing.")
    parser.add_argument("--chunks-json", default=CHUNKS_JSON_PATH)
    parser.add_argument("--namespace", default="", help="Pinecone namespace to patch, e.g. v2_5k or v1_300. Defaults to the index's default namespace.")
    args = parser.parse_args()

    load_dotenv()

    with open(args.chunks_json, encoding="utf-8") as f:
        chunks = json.load(f)

    already_tagged = sum(1 for c in chunks if c.get("metadata", {}).get("source_type") == "evidence")
    print(f"Loaded {len(chunks)} chunks from {args.chunks_json} ({already_tagged} already tagged source_type=evidence locally).")

    ids = [f"{c['metadata']['pmid']}_{c['metadata']['chunk_index']}" for c in chunks]

    if args.dry_run:
        print(f"Dry run — would re-upsert {len(ids)} vectors into namespace {args.namespace!r} with source_type=evidence set (showing first 5 IDs):")
        for i in ids[:5]:
            print(f"  {i}")
        print(f"  ... and {max(0, len(ids) - 5)} more.")
        return

    for c in chunks:
        c["metadata"]["source_type"] = "evidence"

    print(f"Embedding and re-upserting {len(chunks)} chunks into namespace {args.namespace!r} with source_type=evidence (stable IDs — safe to re-run)...")
    upload_chunks(chunks, ids=ids, namespace=args.namespace)
    print(f"Done. Backfilled source_type=evidence onto {len(chunks)} vectors in namespace {args.namespace!r}.")


if __name__ == "__main__":
    main()
