"""
One-time migration: tag every existing vector with source_type="evidence".

Needed because the background tier introduces source_type as a distinguishing
field (evidence vs background), but the 319 PubMed-derived vectors already in
Pinecone predate that field entirely. Uses Pinecone's metadata-only update
(no re-embedding, no re-upsert of vector values) against the exact IDs
already on record locally in data/pubmed_chunks.json — no need to list
vectors from the index itself.

Usage:
    python -m scripts.backfill_source_type --dry-run   # print what would change, touch nothing
    python -m scripts.backfill_source_type             # actually apply
"""
import argparse
import json

from dotenv import load_dotenv

from src.ingest.config import CHUNKS_JSON_PATH
from src.rag.retriever import get_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill source_type=evidence onto existing Pinecone vectors.")
    parser.add_argument("--dry-run", action="store_true", help="Print the IDs that would be updated, without touching Pinecone.")
    parser.add_argument("--chunks-json", default=CHUNKS_JSON_PATH)
    args = parser.parse_args()

    load_dotenv()

    with open(args.chunks_json, encoding="utf-8") as f:
        chunks = json.load(f)

    ids = [f"{c['metadata']['pmid']}_{c['metadata']['chunk_index']}" for c in chunks]
    print(f"Loaded {len(ids)} chunk IDs from {args.chunks_json}.")

    if args.dry_run:
        print("Dry run — would set source_type=evidence on these IDs (showing first 5):")
        for i in ids[:5]:
            print(f"  {i}")
        print(f"  ... and {max(0, len(ids) - 5)} more.")
        return

    index = get_index()
    updated = 0
    for chunk_id in ids:
        index.update(id=chunk_id, set_metadata={"source_type": "evidence"})
        updated += 1
        if updated % 50 == 0:
            print(f"  updated {updated}/{len(ids)}...")

    print(f"Done. Backfilled source_type=evidence onto {updated} vectors.")


if __name__ == "__main__":
    main()
