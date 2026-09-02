"""
Local, free embedding model shared by the ingestion pipeline and the query-time
retriever, so both sides of the RAG system live in the same vector space.
"""
from typing import List

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model_cache = {}


def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


class LocalSentenceTransformerEmbeddings:
    """
    Thin wrapper around a local sentence-transformers model. Runs on CPU, no
    API key or network call required at query/embed time (only the first
    load fetches model weights from the HuggingFace Hub).
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = _get_model(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self._model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return vector.tolist()
