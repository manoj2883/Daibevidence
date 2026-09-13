import csv
import os
from typing import Dict, List

from src.ingest.config import (
    BACKGROUND_CHUNK_OVERLAP_WORDS,
    BACKGROUND_CHUNK_SIZE_WORDS,
    BACKGROUND_CHUNKS_CSV_PATH,
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    CHUNKS_CSV_PATH,
)


def split_text_by_words(text: str, chunk_size_words: int = 300, overlap_words: int = 50) -> List[str]:
    """
    Split a string into overlapping chunks measured in whole words (a sliding
    window over text.split()), rather than characters.
    """
    words = text.split()
    if not words:
        return []

    step = max(chunk_size_words - overlap_words, 1)
    chunks = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size_words]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size_words >= len(words):
            break

    return chunks


CHUNK_METADATA_FIELDS = ["pmid", "title", "journal", "year", "publication_type", "population"]


def split_pubmed_documents(
    documents: List[Dict],
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> List[Dict]:
    """
    Chunk PubMed abstract dicts (as produced by src.ingest.pubmed) into
    word-based chunks, attaching pmid/title/journal/year/publication_type/
    population/chunk_index as metadata on every chunk.
    """
    chunked_docs = []
    for doc in documents:
        abstract = doc.get("abstract", "")
        chunks = split_text_by_words(abstract, chunk_size_words=chunk_size_words, overlap_words=overlap_words)

        for i, chunk in enumerate(chunks):
            metadata = {field: doc.get(field, "") for field in CHUNK_METADATA_FIELDS}
            metadata["chunk_index"] = i
            chunked_docs.append({
                "text": chunk,
                "metadata": metadata,
            })

    return chunked_docs


SECTIONED_CHUNK_METADATA_FIELDS = CHUNK_METADATA_FIELDS + ["population_confidence"]


def split_pubmed_documents_sectioned(
    documents: List[Dict],
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> List[Dict]:
    """
    Phase 3 section-aware chunking: split within each abstract *section*
    (Background/Methods/Results/Conclusions, from doc["sections"], as
    produced by src.ingest.pubmed._parse_article) rather than sliding a
    word-window across the whole abstract regardless of structure — a
    chunk can no longer straddle e.g. the Methods/Results boundary. A
    short section becomes exactly one chunk; a long one is still split
    with the same sliding window, but never crosses into the next
    section. An unstructured abstract (doc["sections"] absent, or a
    single section with label "") degrades to the same behavior as
    split_pubmed_documents.

    Each chunk's metadata records "section" (the label, "" if none) in
    addition to the usual pmid/title/journal/year/publication_type/
    population/chunk_index fields, plus population_confidence (Phase 3
    two-pass tagging).
    """
    chunked_docs = []
    for doc in documents:
        sections = doc.get("sections") or [{"label": "", "text": doc.get("abstract", "")}]
        chunk_index = 0
        for section in sections:
            section_text = section.get("text", "")
            pieces = split_text_by_words(section_text, chunk_size_words=chunk_size_words, overlap_words=overlap_words)
            for piece in pieces:
                metadata = {field: doc.get(field, "") for field in SECTIONED_CHUNK_METADATA_FIELDS}
                metadata["chunk_index"] = chunk_index
                metadata["section"] = section.get("label", "")
                chunked_docs.append({"text": piece, "metadata": metadata})
                chunk_index += 1

    return chunked_docs


def write_sectioned_chunks_csv(chunks: List[Dict], path: str) -> None:
    """
    Export section-aware chunks (with the extra "section" and
    "population_confidence" metadata fields) to CSV.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = SECTIONED_CHUNK_METADATA_FIELDS + ["section", "chunk_index", "text"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for chunk in chunks:
            row = dict(chunk["metadata"])
            row["text"] = chunk["text"]
            writer.writerow(row)


def write_chunks_csv(chunks: List[Dict], path: str = CHUNKS_CSV_PATH) -> None:
    """
    Export chunks (with their full metadata) to CSV for spreadsheet use.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = CHUNK_METADATA_FIELDS + ["chunk_index", "text"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for chunk in chunks:
            row = dict(chunk["metadata"])
            row["text"] = chunk["text"]
            writer.writerow(row)


# --- Background tier (patient-education reference layer) -------------------
# A small, separate class of source alongside the PubMed evidence: definitions,
# diagnostic criteria, and general mechanism from ADA/NIDDK/CDC patient pages.
# Chunked smaller than evidence (these pages are already dense per paragraph;
# smaller chunks give better retrieval granularity — e.g. a "symptoms"
# question shouldn't have to pull in an entire page's worth of unrelated
# "risk factors" and "management" text as one blob).

BACKGROUND_METADATA_FIELDS = ["source_type", "publisher", "title", "source_url", "population"]


def split_background_documents(
    documents: List[Dict],
    chunk_size_words: int = BACKGROUND_CHUNK_SIZE_WORDS,
    overlap_words: int = BACKGROUND_CHUNK_OVERLAP_WORDS,
) -> List[Dict]:
    """
    Chunk background patient-education documents (publisher/title/source_url/
    population/text dicts) into word-based chunks, tagging every chunk
    source_type="background" alongside the usual population/chunk_index.
    """
    chunked_docs = []
    for doc in documents:
        text = doc.get("text", "")
        chunks = split_text_by_words(text, chunk_size_words=chunk_size_words, overlap_words=overlap_words)

        for i, chunk in enumerate(chunks):
            metadata = {field: doc.get(field, "") for field in BACKGROUND_METADATA_FIELDS if field != "source_type"}
            metadata["source_type"] = "background"
            metadata["chunk_index"] = i
            chunked_docs.append({
                "text": chunk,
                "metadata": metadata,
            })

    return chunked_docs


def write_background_chunks_csv(chunks: List[Dict], path: str = BACKGROUND_CHUNKS_CSV_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = BACKGROUND_METADATA_FIELDS + ["chunk_index", "text"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for chunk in chunks:
            row = dict(chunk["metadata"])
            row["text"] = chunk["text"]
            writer.writerow(row)
