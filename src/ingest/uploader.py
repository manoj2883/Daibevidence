import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from src.ingest.local_embeddings import EMBEDDING_DIM, LocalSentenceTransformerEmbeddings

# Load environment variables
load_dotenv()

# The metadata key each chunk's raw text is stored under, alongside its
# pmid/title/journal/year/publication_type/population/chunk_index fields.
TEXT_METADATA_KEY = "text"

# Pinecone's upsert API takes vectors in batches; this keeps individual
# requests small regardless of corpus size.
UPSERT_BATCH_SIZE = 100


def get_embeddings_model() -> LocalSentenceTransformerEmbeddings:
    """
    Initialize the local, free embedding model used for both ingestion and retrieval.
    """
    return LocalSentenceTransformerEmbeddings()


def _ensure_index(pc: Pinecone, index_name: str) -> None:
    """
    Create the Pinecone index if it doesn't exist yet (serverless, cosine metric,
    dimension matching the local embedding model). If it already exists with a
    different dimension, fail loudly instead of silently corrupting retrieval.
    """
    existing = {index.name: index for index in pc.list_indexes()}

    if index_name not in existing:
        cloud = os.environ.get("PINECONE_CLOUD", "aws")
        region = os.environ.get("PINECONE_REGION", "us-east-1")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        return

    existing_dim = existing[index_name].dimension
    if existing_dim != EMBEDDING_DIM:
        raise ValueError(
            f"Index '{index_name}' already exists with dimension {existing_dim}, but the "
            f"local embedding model produces {EMBEDDING_DIM}-dim vectors. Use a different "
            f"index name or delete/recreate the existing index."
        )


def upload_chunks(
    chunks: List[Dict[str, Any]],
    index_name: str = None,
    ids: Optional[List[str]] = None,
) -> None:
    """
    Embed and upsert chunks into Pinecone. Each chunk should be a dict with
    'text' and optional 'metadata'. Pass `ids` (one per chunk) for a stable,
    idempotent upsert — re-running with the same ids overwrites rather than
    duplicates.

    Stores each chunk's text under the "text" metadata key (matching the
    convention already live on the existing index) so retrieval can read it
    straight back out of the returned metadata.
    """
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY must be set in environment variables.")
    pinecone_key = pinecone_key.strip()

    index_name = index_name or os.environ.get("PINECONE_INDEX_NAME")
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME must be set in environment variables.")
    index_name = index_name.strip()

    pc = Pinecone(api_key=pinecone_key)
    _ensure_index(pc, index_name)
    index = pc.Index(index_name)

    embeddings = get_embeddings_model()

    texts = [c["text"] for c in chunks]
    metadatas = [c.get("metadata", {}) for c in chunks]
    if ids is None:
        ids = [str(i) for i in range(len(chunks))]

    vectors = embeddings.embed_documents(texts)

    records = []
    for chunk_id, vector, text, metadata in zip(ids, vectors, texts, metadatas):
        record_metadata = {**metadata, TEXT_METADATA_KEY: text}
        records.append({"id": chunk_id, "values": vector, "metadata": record_metadata})

    for start in range(0, len(records), UPSERT_BATCH_SIZE):
        index.upsert(vectors=records[start : start + UPSERT_BATCH_SIZE])
