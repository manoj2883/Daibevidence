"""
Ingestion pipeline entrypoint: pull diabetes abstracts across four topics
(diet & nutrition, glycemic control, body composition & weight, diet/medication
interaction), cache raw + chunked data to JSON and CSV, then embed locally and
upsert into Pinecone with stable, duplicate-safe IDs.

Scope, filters, and limits are controlled by src.ingest.config.

Usage:
    python -m src.ingest.run_ingest
    python -m src.ingest.run_ingest --skip-fetch          # reuse data/pubmed_raw.json
    python -m src.ingest.run_ingest --fetch-limit 100
    python -m src.ingest.run_ingest --dry-run              # fetch/chunk/export, skip Pinecone
"""
import argparse
import json
import os

from dotenv import load_dotenv

from src.ingest.chunker import split_pubmed_documents, write_chunks_csv
from src.ingest.config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    CHUNKS_CSV_PATH,
    CHUNKS_JSON_PATH,
    CORPUS_MANIFEST_PATH,
    FETCH_LIMIT,
    FETCH_META_PATH,
    RAW_CSV_PATH,
    RAW_JSON_PATH,
)
from src.ingest.freeze import write_corpus_manifest
from src.ingest.pubmed import load_cached_pubmed, pull_and_cache_pubmed
from src.ingest.uploader import upload_chunks


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Pull diabetes abstracts from PubMed and ingest them into Pinecone.")
    parser.add_argument("--fetch-limit", type=int, default=FETCH_LIMIT, help="Max total abstracts to fetch.")
    parser.add_argument("--raw-json", default=RAW_JSON_PATH, help="Path to the raw abstract JSON cache.")
    parser.add_argument("--raw-csv", default=RAW_CSV_PATH, help="Path to the raw abstract CSV export.")
    parser.add_argument("--chunks-json", default=CHUNKS_JSON_PATH, help="Path to the processed-chunk JSON cache.")
    parser.add_argument("--chunks-csv", default=CHUNKS_CSV_PATH, help="Path to the processed-chunk CSV export.")
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Load abstracts from --raw-json instead of hitting the PubMed API again.",
    )
    parser.add_argument("--chunk-size-words", type=int, default=CHUNK_SIZE_WORDS)
    parser.add_argument("--overlap-words", type=int, default=CHUNK_OVERLAP_WORDS)
    parser.add_argument("--dry-run", action="store_true", help="Fetch/chunk/export but skip the Pinecone upload.")
    parser.add_argument("--no-manifest", action="store_true", help="Skip writing the corpus freeze manifest.")
    args = parser.parse_args()

    if args.skip_fetch:
        if not os.path.exists(args.raw_json):
            raise SystemExit(f"--skip-fetch was set but {args.raw_json} does not exist. Run without it first.")
        print(f"Loading cached abstracts from {args.raw_json}...")
        documents = load_cached_pubmed(args.raw_json)
    else:
        print(f"Querying PubMed across 4 topics for up to {args.fetch_limit} abstracts...")
        documents = pull_and_cache_pubmed(
            fetch_limit=args.fetch_limit, out_json=args.raw_json, out_csv=args.raw_csv
        )
        print(f"Cached {len(documents)} raw abstracts to {args.raw_json} and {args.raw_csv}.")

    print(f"Fetched {len(documents)} abstracts with text.")
    if not documents:
        print("No documents available — nothing to ingest.")
        return

    # Raw abstracts are saved above (or already on disk via --skip-fetch) before
    # any chunking or embedding happens.

    chunks = split_pubmed_documents(
        documents, chunk_size_words=args.chunk_size_words, overlap_words=args.overlap_words
    )
    print(f"Created {len(chunks)} chunks.")

    # Save processed chunks (with full metadata) to JSON + CSV before anything
    # is sent to Pinecone.
    os.makedirs(os.path.dirname(args.chunks_json) or ".", exist_ok=True)
    with open(args.chunks_json, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    write_chunks_csv(chunks, args.chunks_csv)
    print(f"Saved processed chunks to {args.chunks_json} and {args.chunks_csv}.")

    uploaded = False
    if args.dry_run:
        print("Dry run requested — skipping Pinecone upload.")
    else:
        ids = [f"{c['metadata']['pmid']}_{c['metadata']['chunk_index']}" for c in chunks]
        print("Embedding and upserting chunks into Pinecone (stable IDs — safe to re-run)...")
        upload_chunks(chunks, ids=ids)
        uploaded = True
        print("Ingestion complete.")

    if not args.no_manifest:
        manifest = write_corpus_manifest(
            documents,
            chunks,
            uploaded_to_pinecone=uploaded,
            pinecone_index_name=os.environ.get("PINECONE_INDEX_NAME"),
        )
        print(f"Wrote corpus manifest to {CORPUS_MANIFEST_PATH} (fetched_at: {manifest['fetch'].get('fetched_at', 'unknown')}).")


if __name__ == "__main__":
    main()
