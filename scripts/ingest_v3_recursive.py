"""
Builds the v3_5k_recursive namespace: the same 5,000 v2_5k abstracts (minus any
tagged "Retracted Publication") plus the background pages, re-chunked with
src.ingest.recursive_chunker and embedded with all-MiniLM-L6-v2 at 256 tokens.

Publication type is already stored per abstract in data/v2_5k_raw.json, so no
E-utilities re-fetch is needed; the script refuses to run if any abstract is
missing it.

Also writes corpus/v3_5k_recursive.json.gz (tracked in git): every chunk plus
every full parent document. The query path needs it at runtime for BM25 and to
send the full parent abstract to the judge and generation.

v1_300 and v2_5k are never touched.

Usage:
    python -m scripts.ingest_v3_recursive              # chunk, write corpus file, upload
    python -m scripts.ingest_v3_recursive --dry-run    # chunk + corpus file only
"""
import argparse
import gzip
import json
import os
from collections import Counter
from datetime import datetime, timezone

from dotenv import load_dotenv

from src.ingest.config import BACKGROUND_RAW_JSON_PATH, NAMESPACE_V3, V2_5K_RAW_JSON_PATH, V3_CORPUS_PATH, V3_MANIFEST_PATH
from src.ingest.local_embeddings import EMBEDDING_MAX_TOKENS, MODEL_NAME
from src.ingest.recursive_chunker import (
    MAX_CHUNK_TOKENS,
    background_parent_id,
    is_retracted,
    parent_text,
    split_background_documents_recursive,
    split_pubmed_documents_recursive,
    study_design,
)
from src.ingest.uploader import upload_chunks


def build_parents(documents, background):
    parents = {}
    for doc in documents:
        if is_retracted(doc.get("publication_type", "")):
            continue
        parents[doc["pmid"]] = {
            "source_type": "evidence",
            "pmid": doc["pmid"],
            "title": doc.get("title", ""),
            "journal": doc.get("journal", ""),
            "year": doc.get("year", ""),
            "publication_type": doc.get("publication_type", ""),
            "study_design": study_design(doc.get("publication_type", "")),
            "population": doc.get("population", ""),
            "text": parent_text(doc),
        }
    for doc in background:
        pid = background_parent_id(doc["publisher"], doc["title"])
        parents[pid] = {
            "source_type": "background",
            "publisher": doc["publisher"],
            "title": doc["title"],
            "source_url": doc.get("source_url", ""),
            "population": doc.get("population", ""),
            "publication_type": "Background",
            "study_design": f"Background ({doc['publisher']})",
            "text": parent_text(doc),
        }
    return parents


def main():
    parser = argparse.ArgumentParser(description="Re-chunk v2_5k + background into the v3_5k_recursive namespace.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--namespace", default=NAMESPACE_V3)
    args = parser.parse_args()
    load_dotenv()

    with open(V2_5K_RAW_JSON_PATH, encoding="utf-8") as f:
        documents = json.load(f)
    with open(BACKGROUND_RAW_JSON_PATH, encoding="utf-8") as f:
        background = json.load(f)

    missing = [d["pmid"] for d in documents if not d.get("publication_type")]
    if missing:
        raise SystemExit(f"{len(missing)} abstracts have no stored publication type (e.g. {missing[:5]}); fetch them first.")
    retracted = sorted(d["pmid"] for d in documents if is_retracted(d.get("publication_type", "")))

    chunks = split_pubmed_documents_recursive(documents) + split_background_documents_recursive(background)
    ids = [f"{c['metadata']['parent_id']}_{c['metadata']['chunk_index']}" for c in chunks]
    parents = build_parents(documents, background)
    print(f"{len(chunks)} chunks from {len(parents)} parents ({len(retracted)} retracted abstracts excluded: {retracted}).")

    os.makedirs(os.path.dirname(V3_CORPUS_PATH), exist_ok=True)
    corpus = {
        "namespace": args.namespace,
        "chunks": [{"id": i, "text": c["text"], "metadata": c["metadata"]} for i, c in zip(ids, chunks)],
        "parents": parents,
    }
    with gzip.open(V3_CORPUS_PATH, "wt", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)
    print(f"Wrote {V3_CORPUS_PATH} ({os.path.getsize(V3_CORPUS_PATH) / 1e6:.1f} MB).")

    uploaded = False
    if not args.dry_run:
        print(f"Embedding and upserting into namespace {args.namespace!r} (stable ids, safe to re-run)...")
        batch = 1000
        for start in range(0, len(chunks), batch):
            upload_chunks(chunks[start:start + batch], ids=ids[start:start + batch], namespace=args.namespace)
            print(f"  {min(start + batch, len(chunks))}/{len(chunks)}")
        uploaded = True

    manifest = {
        "namespace": args.namespace,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_raw": [V2_5K_RAW_JSON_PATH, BACKGROUND_RAW_JSON_PATH],
        "splitter": "src.ingest.recursive_chunker (sections -> sentences -> words, last-sentence overlap)",
        "max_chunk_tokens_with_header": MAX_CHUNK_TOKENS,
        "embedding_model": MODEL_NAME,
        "embedding_max_tokens": EMBEDDING_MAX_TOKENS,
        "chunks": len(chunks),
        "evidence_chunks": sum(1 for c in chunks if c["metadata"]["source_type"] == "evidence"),
        "background_chunks": sum(1 for c in chunks if c["metadata"]["source_type"] == "background"),
        "parents": len(parents),
        "retracted_excluded": retracted,
        "study_design_counts_by_abstract": dict(Counter(p["study_design"] for p in parents.values())),
        "uploaded_to_pinecone": uploaded,
    }
    with open(V3_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {V3_MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
