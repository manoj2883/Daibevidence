"""
Phase 3 corpus expansion: fetch ~5,000 PubMed abstracts under the exact
same scope as the original 300-abstract corpus (DIABETES_SCOPE, TOPICS,
PUBLICATION_TYPES, DATE_RANGE_YEARS — only FETCH_LIMIT changes), section-
aware chunk them, two-pass population-tag them (already done at parse
time by src.ingest.pubmed._parse_article), and upsert into a separate
Pinecone namespace ("v2_5k") so it can be compared side-by-side against
the original corpus (namespace "v1_300") without touching it.

Verified live against PubMed before writing this script: the same scope
that produced the 300-abstract corpus has ~25,000 matching abstracts
pre-dedup, comfortably more than 5,000 — so v2_5k stays a clean "same
scope, more data" comparison, not a scope change bundled in as well.

Checkpointed per topic: each of the 4 topics fetches up to ~1,250 PMIDs
(several efetch calls), and completed topics are saved to
V2_5K_FETCH_CHECKPOINT_PATH as they finish. --resume skips any topic
already in the checkpoint, so a network drop mid-fetch loses at most the
in-progress topic, not the whole run.

Usage:
    python -m scripts.fetch_corpus                      # fetch, chunk, upload to Pinecone namespace v2_5k
    python -m scripts.fetch_corpus --resume              # continue after an interruption
    python -m scripts.fetch_corpus --dry-run             # fetch/chunk/export only, skip Pinecone
    python -m scripts.fetch_corpus --skip-fetch          # reuse the cached raw JSON, re-chunk/re-upload only
"""
import argparse
import json
import os

from dotenv import load_dotenv

from src.ingest.chunker import split_pubmed_documents_sectioned, write_sectioned_chunks_csv
from src.ingest.config import (
    NAMESPACE_V2_5K,
    TOPICS,
    V2_5K_CHUNKS_CSV_PATH,
    V2_5K_CHUNKS_JSON_PATH,
    V2_5K_FETCH_CHECKPOINT_PATH,
    V2_5K_FETCH_LIMIT,
    V2_5K_FETCH_META_PATH,
    V2_5K_MANIFEST_PATH,
    V2_5K_RAW_CSV_PATH,
    V2_5K_RAW_JSON_PATH,
)
from src.ingest.freeze import write_corpus_manifest
from src.ingest.pubmed import (
    _date_range_strings,
    _write_raw_csv,
    build_topic_query,
    fetch_pubmed_abstracts,
    search_pubmed,
)
from src.ingest.uploader import upload_chunks


def load_checkpoint(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(path, checkpoint):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def fetch_checkpointed(fetch_limit, checkpoint_path, resume):
    """
    Same per-topic search+fetch logic as src.ingest.pubmed.pull_and_cache_pubmed,
    but with a topic-granularity checkpoint so a network drop loses at most
    one topic's progress. Returns (documents, queries_run, duplicate_pmids_skipped).
    """
    checkpoint = load_checkpoint(checkpoint_path) if resume else {}
    if checkpoint:
        done_counts = {k: len(v["documents"]) for k, v in checkpoint.items()}
        print(f"Resuming: topics already fetched: {done_counts}")

    mindate, maxdate = _date_range_strings()
    per_topic_max = max(fetch_limit // len(TOPICS), 1)

    seen_pmids = set()
    for entry in checkpoint.values():
        seen_pmids.update(doc["pmid"] for doc in entry["documents"])

    # Top-up entries (keyed "{topic}__topup") don't carry their own "query"
    # field — they reuse the main topic's query — so only the original
    # per-topic entries feed queries_run.
    queries_run = {k: v["query"] for k, v in checkpoint.items() if k in TOPICS}
    duplicate_pmids_skipped = sum(v.get("duplicate_pmids_skipped", 0) for v in checkpoint.values())
    all_documents = [doc for entry in checkpoint.values() for doc in entry["documents"]]

    for topic_key, topic_fragment in TOPICS.items():
        if topic_key in checkpoint:
            continue

        remaining = fetch_limit - len(all_documents)
        if remaining <= 0:
            break

        query = build_topic_query(topic_fragment)
        retmax = min(per_topic_max, remaining)
        print(f"[{topic_key}] searching (retmax={retmax})...")
        pmids = search_pubmed(query, retmax=retmax, mindate=mindate, maxdate=maxdate)
        unique_pmids = [p for p in pmids if p not in seen_pmids]
        topic_duplicates = len(pmids) - len(unique_pmids)
        new_pmids = unique_pmids[:remaining]
        seen_pmids.update(new_pmids)

        print(f"[{topic_key}] fetching {len(new_pmids)} abstracts...")
        docs = fetch_pubmed_abstracts(new_pmids) if new_pmids else []
        for doc in docs:
            doc["topic"] = topic_key

        checkpoint[topic_key] = {
            "query": query,
            "documents": docs,
            "duplicate_pmids_skipped": topic_duplicates,
        }
        save_checkpoint(checkpoint_path, checkpoint)
        print(f"[{topic_key}] done: {len(docs)} abstracts (checkpoint saved).")

        queries_run[topic_key] = query
        duplicate_pmids_skipped += topic_duplicates
        all_documents.extend(docs)

    # Top-up pass: the even per-topic split (fetch_limit // len(TOPICS))
    # can leave a shortfall when one topic simply has fewer matching
    # abstracts than its even share (e.g. diet_medication_interaction is
    # a much narrower MeSH combination than the other three topics) —
    # redistribute that unused capacity to whichever topics still have
    # supply beyond what they already contributed, instead of silently
    # returning fewer abstracts than requested.
    shortfall = fetch_limit - len(all_documents)
    if shortfall > 0:
        print(f"Shortfall of {shortfall} after the even per-topic split — topping up from topics with remaining supply...")
    for topic_key in TOPICS:
        if shortfall <= 0:
            break
        topup_key = f"{topic_key}__topup"
        if topup_key in checkpoint:
            entry = checkpoint[topup_key]
        else:
            query = queries_run[topic_key]
            retmax = per_topic_max + shortfall + 200  # buffer for expected dedup/no-abstract drops
            pmids = search_pubmed(query, retmax=retmax, mindate=mindate, maxdate=maxdate)
            unique_pmids = [p for p in pmids if p not in seen_pmids]
            new_pmids = unique_pmids[:shortfall]
            seen_pmids.update(new_pmids)
            docs = fetch_pubmed_abstracts(new_pmids) if new_pmids else []
            for doc in docs:
                doc["topic"] = topic_key
            entry = {"documents": docs, "duplicate_pmids_skipped": len(pmids) - len(unique_pmids)}
            checkpoint[topup_key] = entry
            save_checkpoint(checkpoint_path, checkpoint)

        duplicate_pmids_skipped += entry.get("duplicate_pmids_skipped", 0)
        all_documents.extend(entry["documents"])
        shortfall = fetch_limit - len(all_documents)
        print(f"[{topic_key} top-up] contributed {len(entry['documents'])} more abstracts (shortfall now {max(shortfall, 0)}).")

    all_documents = all_documents[:fetch_limit]
    return all_documents, queries_run, duplicate_pmids_skipped


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Fetch the Phase 3 v2_5k corpus expansion and upload it to its own Pinecone namespace.")
    parser.add_argument("--fetch-limit", type=int, default=V2_5K_FETCH_LIMIT)
    parser.add_argument("--resume", action="store_true", help="Skip topics already saved in the fetch checkpoint.")
    parser.add_argument("--skip-fetch", action="store_true", help="Reuse the cached raw JSON instead of hitting PubMed again.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch/chunk/export but skip the Pinecone upload.")
    parser.add_argument("--namespace", default=NAMESPACE_V2_5K)
    args = parser.parse_args()

    if args.skip_fetch:
        if not os.path.exists(V2_5K_RAW_JSON_PATH):
            raise SystemExit(f"--skip-fetch was set but {V2_5K_RAW_JSON_PATH} does not exist. Run without it first.")
        print(f"Loading cached abstracts from {V2_5K_RAW_JSON_PATH}...")
        with open(V2_5K_RAW_JSON_PATH, encoding="utf-8") as f:
            documents = json.load(f)
        duplicate_pmids_skipped = None
    else:
        print(f"Fetching up to {args.fetch_limit} abstracts across {len(TOPICS)} topics (checkpointed)...")
        documents, queries_run, duplicate_pmids_skipped = fetch_checkpointed(
            args.fetch_limit, V2_5K_FETCH_CHECKPOINT_PATH, args.resume
        )

        os.makedirs(os.path.dirname(V2_5K_RAW_JSON_PATH) or ".", exist_ok=True)
        with open(V2_5K_RAW_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        _write_raw_csv(documents, V2_5K_RAW_CSV_PATH)

        fetch_meta = {
            "fetch_limit_requested": args.fetch_limit,
            "abstracts_fetched": len(documents),
            "topic_queries": queries_run,
            "duplicate_pmids_skipped": duplicate_pmids_skipped,
            "pmids": [doc["pmid"] for doc in documents],
        }
        os.makedirs(os.path.dirname(V2_5K_FETCH_META_PATH) or ".", exist_ok=True)
        with open(V2_5K_FETCH_META_PATH, "w", encoding="utf-8") as f:
            json.dump(fetch_meta, f, ensure_ascii=False, indent=2)

        print(f"Fetched {len(documents)} abstracts ({duplicate_pmids_skipped} duplicate PMIDs skipped across topic queries).")

    if not documents:
        print("No documents available — nothing to ingest.")
        return

    print("Section-aware chunking...")
    chunks = split_pubmed_documents_sectioned(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} abstracts.")

    os.makedirs(os.path.dirname(V2_5K_CHUNKS_JSON_PATH) or ".", exist_ok=True)
    with open(V2_5K_CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    write_sectioned_chunks_csv(chunks, V2_5K_CHUNKS_CSV_PATH)
    print(f"Saved chunks to {V2_5K_CHUNKS_JSON_PATH} and {V2_5K_CHUNKS_CSV_PATH}.")

    uploaded = False
    if args.dry_run:
        print("Dry run requested — skipping Pinecone upload.")
    else:
        ids = [f"{c['metadata']['pmid']}_{c['metadata']['chunk_index']}" for c in chunks]
        print(f"Embedding and upserting {len(chunks)} chunks into Pinecone namespace {args.namespace!r} (stable IDs — safe to re-run)...")
        upload_chunks(chunks, ids=ids, namespace=args.namespace)
        uploaded = True
        print("Upload complete.")

    manifest = write_corpus_manifest(
        documents,
        chunks,
        uploaded_to_pinecone=uploaded,
        pinecone_index_name=os.environ.get("PINECONE_INDEX_NAME"),
        pinecone_namespace=args.namespace,
        fetch_meta_path=V2_5K_FETCH_META_PATH,
        out_path=V2_5K_MANIFEST_PATH,
    )
    print(f"Wrote corpus manifest to {V2_5K_MANIFEST_PATH}.")
    print(json.dumps(manifest["corpus_stats"], indent=2))


if __name__ == "__main__":
    main()
