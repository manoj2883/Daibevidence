"""
Background-tier ingestion entrypoint: chunk the curated ADA/NIDDK/CDC
patient-education documents (src.ingest.background_sources), cache raw +
chunked data to JSON and CSV, then embed locally and upsert into the same
Pinecone index as the evidence corpus, tagged source_type="background".

Usage:
    python -m src.ingest.run_background_ingest
    python -m src.ingest.run_background_ingest --dry-run   # chunk/export, skip Pinecone
"""
import argparse
import csv
import json
import os
import re

from dotenv import load_dotenv

from src.ingest.background_sources import BACKGROUND_DOCUMENTS
from src.ingest.chunker import split_background_documents, write_background_chunks_csv
from src.ingest.config import (
    BACKGROUND_CHUNK_OVERLAP_WORDS,
    BACKGROUND_CHUNK_SIZE_WORDS,
    BACKGROUND_CHUNKS_CSV_PATH,
    BACKGROUND_CHUNKS_JSON_PATH,
    BACKGROUND_RAW_CSV_PATH,
    BACKGROUND_RAW_JSON_PATH,
)
from src.ingest.uploader import upload_chunks


def _slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len]


def _stable_id(publisher: str, title: str, chunk_index: int) -> str:
    return f"bg_{_slug(publisher)}_{_slug(title)}_{chunk_index}"


def _write_raw_csv(documents, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = ["publisher", "title", "source_url", "population", "text"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for doc in documents:
            writer.writerow({k: doc.get(k, "") for k in fieldnames})


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Chunk and ingest the background (patient-education) tier into Pinecone.")
    parser.add_argument("--raw-json", default=BACKGROUND_RAW_JSON_PATH)
    parser.add_argument("--raw-csv", default=BACKGROUND_RAW_CSV_PATH)
    parser.add_argument("--chunks-json", default=BACKGROUND_CHUNKS_JSON_PATH)
    parser.add_argument("--chunks-csv", default=BACKGROUND_CHUNKS_CSV_PATH)
    parser.add_argument("--chunk-size-words", type=int, default=BACKGROUND_CHUNK_SIZE_WORDS)
    parser.add_argument("--overlap-words", type=int, default=BACKGROUND_CHUNK_OVERLAP_WORDS)
    parser.add_argument("--dry-run", action="store_true", help="Chunk/export but skip the Pinecone upload.")
    args = parser.parse_args()

    documents = BACKGROUND_DOCUMENTS
    print(f"{len(documents)} curated background documents.")

    # Save raw source documents before anything else happens.
    os.makedirs(os.path.dirname(args.raw_json) or ".", exist_ok=True)
    with open(args.raw_json, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    _write_raw_csv(documents, args.raw_csv)
    print(f"Saved raw background documents to {args.raw_json} and {args.raw_csv}.")

    chunks = split_background_documents(documents, chunk_size_words=args.chunk_size_words, overlap_words=args.overlap_words)
    print(f"Created {len(chunks)} background chunks.")

    with open(args.chunks_json, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    write_background_chunks_csv(chunks, args.chunks_csv)
    print(f"Saved processed background chunks to {args.chunks_json} and {args.chunks_csv}.")

    if args.dry_run:
        print("Dry run requested — skipping Pinecone upload.")
        return

    ids = [_stable_id(c["metadata"]["publisher"], c["metadata"]["title"], c["metadata"]["chunk_index"]) for c in chunks]
    print("Embedding and upserting background chunks into Pinecone (stable IDs — safe to re-run)...")
    upload_chunks(chunks, ids=ids)
    print("Background ingestion complete.")


if __name__ == "__main__":
    main()
