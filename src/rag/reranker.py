"""
Local cross-encoder reranking over an already-retrieved candidate pool.
Same fastembed/ONNX approach as src.ingest.local_embeddings (no torch, no
API key, no network call once weights are cached) — a query-passage joint
scorer rather than a bi-encoder cosine comparison, tried here specifically
because retrieval similarity alone cannot separate in-scope questions from
diabetes-adjacent-but-off-topic ones (see eval/eval_set_v2.json and
data/eval_v2_rerank_*.json).
"""
from typing import List, Tuple

from fastembed.rerank.cross_encoder import TextCrossEncoder

MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"

_model_cache = {}


def _get_model(model_name: str) -> TextCrossEncoder:
    if model_name not in _model_cache:
        _model_cache[model_name] = TextCrossEncoder(model_name=model_name)
    return _model_cache[model_name]


def rerank_scores(query: str, documents: List[str], model_name: str = MODEL_NAME) -> List[float]:
    """
    Raw cross-encoder logits, one per document, in the same order as
    `documents` (fastembed's rerank() does not sort). Not bounded to
    [0, 1] and not comparable in absolute terms to cosine similarity —
    only the relative separation between groups matters here.
    """
    if not documents:
        return []
    model = _get_model(model_name)
    return list(model.rerank(query, documents))


def rerank(query: str, documents: List[str], model_name: str = MODEL_NAME) -> List[Tuple[int, float]]:
    """
    (original_index, score) pairs sorted descending by score — the
    candidate pool reordered by cross-encoder relevance.
    """
    scores = rerank_scores(query, documents, model_name=model_name)
    return sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
