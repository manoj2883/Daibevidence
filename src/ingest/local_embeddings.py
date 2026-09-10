"""
Local, free embedding model shared by the ingestion pipeline and the query-time
retriever, so both sides of the RAG system live in the same vector space.

Uses fastembed (pure ONNX runtime, no torch/transformers) instead of
sentence-transformers' PyTorch backend — the torch+transformers stack loads
well over 512MB RAM, which OOM-crashes on Render's free tier. fastembed runs
the same model (~220MB RSS total) and produces numerically identical output
(verified: cosine similarity 1.000000 against the torch backend on sample text).
"""
from typing import List

from fastembed import TextEmbedding

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model_cache = {}


def _get_model(model_name: str) -> TextEmbedding:
    if model_name not in _model_cache:
        _model_cache[model_name] = TextEmbedding(model_name=model_name)
    return _model_cache[model_name]


class LocalSentenceTransformerEmbeddings:
    """
    Thin wrapper around a local ONNX embedding model. Runs on CPU, no API
    key or network call required at query/embed time (only the first load
    fetches model weights from the HuggingFace Hub).
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = _get_model(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return list(self._model.embed([text]))[0].tolist()
