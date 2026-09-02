"""
Prompt-size guardrails. A previous run accidentally sent the whole corpus to
Claude instead of the top-k chunks and burned 15M input tokens — that was a
bug, not expected behaviour. This module gives every call site two things:
a hard cap that fails loudly instead of silently sending too much, and a
rough cost estimate to print before anything that calls the Anthropic API.
"""
from typing import List

from src.rag.types import RetrievedChunk

# ~4 characters per token is a standard rough estimate for English text.
# This is deliberately approximate — good enough to catch a 15M-token bug,
# not a substitute for the API's own usage reporting.
CHARS_PER_TOKEN_ESTIMATE = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def assert_chunk_cap(chunks: List[RetrievedChunk], max_chunks: int) -> None:
    """
    Hard safety cap: raise instead of silently sending more than max_chunks
    to the LLM. A printed warning can be missed in a batch run; this can't.
    """
    if len(chunks) > max_chunks:
        raise ValueError(
            f"Retrieved {len(chunks)} chunks but the cap is {max_chunks} — "
            f"refusing to send this to Claude. This should never happen from "
            f"a normal top-k retrieval call; check the retriever."
        )


def print_prompt_estimate(label: str, *texts: str) -> int:
    """
    Print a rough token/char estimate for a prompt about to be sent to
    Claude. Returns the estimated token count so callers can sum it across
    a batch (e.g. the evaluation harness) before running.
    """
    total_chars = sum(len(t) for t in texts)
    est_tokens = estimate_tokens(" ".join(texts))
    print(f"[cost] {label}: ~{total_chars} chars, ~{est_tokens} tokens (rough estimate)")
    return est_tokens
