"""
Local, free embedding model shared by the ingestion pipeline and the query-time
retriever, so both sides of the RAG system live in the same vector space.

Uses fastembed (pure ONNX runtime, no torch/transformers) instead of
sentence-transformers' PyTorch backend — the torch+transformers stack loads
well over 512MB RAM, which OOM-crashes on Render's free tier. fastembed runs
the same model (~220MB RSS total) and produces numerically identical output
(verified: cosine similarity 1.000000 against the torch backend on sample text).

Truncation: fastembed reads max_length=128 from this model's tokenizer config,
so anything past 128 MiniLM tokens was silently dropped at embed time (40.3% of
v2_5k chunks were longer than that). sentence-transformers trained this model
at 256, so the tokenizer is reset to EMBEDDING_MAX_TOKENS here. Queries are far
shorter than 128 tokens, so query vectors against v1_300/v2_5k are unchanged.
"""
from typing import List

from fastembed import TextEmbedding
from tokenizers import Tokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_MAX_TOKENS = 256

_model_cache = {}
_count_tokenizer_cache = {}


def _get_model(model_name: str) -> TextEmbedding:
    if model_name not in _model_cache:
        model = TextEmbedding(model_name=model_name)
        tokenizer = model.model.tokenizer
        tokenizer.enable_truncation(max_length=EMBEDDING_MAX_TOKENS)
        # The shipped tokenizer.json pads to a fixed 128; pad to the batch's longest instead.
        padding = tokenizer.padding or {}
        tokenizer.enable_padding(pad_id=padding.get("pad_id", 0), pad_token=padding.get("pad_token", "[PAD]"))
        _model_cache[model_name] = model
    return _model_cache[model_name]


def _get_count_tokenizer(model_name: str) -> Tokenizer:
    """A non-truncating copy of the model's tokenizer, so counts are real lengths."""
    if model_name not in _count_tokenizer_cache:
        tok = Tokenizer.from_str(_get_model(model_name).model.tokenizer.to_str())
        tok.no_truncation()
        tok.no_padding()
        _count_tokenizer_cache[model_name] = tok
    return _count_tokenizer_cache[model_name]


def count_tokens(text: str, model_name: str = MODEL_NAME) -> int:
    """MiniLM token count including the [CLS]/[SEP] special tokens."""
    return len(_get_count_tokenizer(model_name).encode(text).ids)


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
