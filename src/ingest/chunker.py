import csv
import os
from typing import Dict, List

from src.ingest.config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS, CHUNKS_CSV_PATH


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
