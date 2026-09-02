import os
from typing import List, Optional

from dotenv import load_dotenv
from pinecone import Pinecone

from src.ingest.config import RETRIEVAL_TOP_K
from src.ingest.local_embeddings import LocalSentenceTransformerEmbeddings
from src.ingest.uploader import TEXT_METADATA_KEY
from src.rag.types import RetrievedChunk

# Load environment variables
load_dotenv()

_index_cache = {}
_embeddings_cache: Optional[LocalSentenceTransformerEmbeddings] = None


def get_index():
    """
    Initialize and return the raw Pinecone index handle.
    """
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY must be set in environment variables.")
    pinecone_key = pinecone_key.strip()

    index_name = os.environ.get("PINECONE_INDEX_NAME")
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME must be set in environment variables.")
    index_name = index_name.strip()

    cache_key = (pinecone_key, index_name)
    if cache_key not in _index_cache:
        pc = Pinecone(api_key=pinecone_key)
        _index_cache[cache_key] = pc.Index(index_name)
    return _index_cache[cache_key]


def get_retriever_k() -> int:
    return int(os.environ.get("RETRIEVER_TOP_K", str(RETRIEVAL_TOP_K)))


def retrieve(question: str, k: Optional[int] = None) -> List[RetrievedChunk]:
    """
    Embed the question locally and query Pinecone for the top-k most similar
    chunks. Returns them ordered by descending similarity score, exactly as
    Pinecone returns them — never more than k.
    """
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = LocalSentenceTransformerEmbeddings()

    k = k or get_retriever_k()
    query_vector = _embeddings_cache.embed_query(question)

    index = get_index()
    response = index.query(vector=query_vector, top_k=k, include_metadata=True)

    chunks = []
    for match in response.get("matches", []):
        metadata = dict(match.get("metadata") or {})
        text = metadata.pop(TEXT_METADATA_KEY, "")
        chunks.append(RetrievedChunk(text=text, metadata=metadata, score=float(match.get("score", 0.0))))
    return chunks
