"""
LLM-based answerability judge, run between retrieval and generation.

docs/findings-retrieval-floor.md established that cosine similarity over
this corpus measures topical relatedness ("is this about diabetes"), not
answerability ("does this specific excerpt answer this specific
question") — a diabetes-adjacent but out-of-scope question (e.g. diabetic
retinopathy laser treatment) can score as high as a genuinely in-scope
one, so no similarity floor can separate them. This module replaces that
threshold decision with a direct judgment from a small model, over the
same excerpts the main model would see.
"""
import json
import logging
import os
import re
from typing import Any, Dict, List

import anthropic

from src.rag.cost import print_prompt_estimate
from src.rag.types import RetrievedChunk

logger = logging.getLogger("diabevidence")

# Deliberately smaller/cheaper than chain.get_model()'s main answer model —
# this call only has to make a binary judgment, not write a cited answer.
JUDGE_MODEL_DEFAULT = "claude-haiku-4-5"

JUDGE_SYSTEM_PROMPT = """You are an answerability judge for a diabetes-research question-answering \
system. You will be given a user's question and the excerpts retrieved for it. Decide whether these \
excerpts contain enough evidence to actually answer THIS specific question.

Topical relatedness is not enough — the excerpts must actually address what is being asked. For \
example, an excerpt about diabetic retinopathy laser treatment is related to diabetes, but it does not \
answer a question about diet and glycemic control, even though it would likely score as "similar" \
under a generic similarity search.

Respond with ONLY a JSON object and nothing else — no markdown code fences, no explanation outside the \
object — in exactly this shape:
{"answerable": true or false, "reason": "<one sentence>"}"""


def get_judge_model() -> str:
    return os.environ.get("ANTHROPIC_JUDGE_MODEL", JUDGE_MODEL_DEFAULT)


def _format_chunks_for_judge(chunks: List[RetrievedChunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata or {}
        source_type = meta.get("source_type", "evidence")
        title = meta.get("title", "")
        if source_type == "background":
            label = meta.get("publisher", "background")
        else:
            label = f"PMID {meta.get('pmid', 'unknown')}"
        blocks.append(f"[Excerpt {i}] ({source_type}, {label}) {title}\n{chunk.text}")
    return "\n\n".join(blocks)


def _strip_code_fences(text: str) -> str:
    """
    A small model asked for "JSON only" still occasionally wraps it in a
    ```json ... ``` fence — strip that before parsing rather than failing
    on otherwise-valid JSON. Only handles the case where the fence brackets
    the *entire* response; see extract_json_object() for the more general
    case (a fence, or bare braces, with prose before and/or after).
    """
    text = text.strip()
    if not text.startswith("```"):
        return text
    text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    first_line, sep, rest = text.partition("\n")
    if sep and first_line.strip().lower() in ("json", ""):
        text = rest
    return text.strip()


def _find_matching_brace(text: str, start: int) -> int:
    """
    Given text[start] == '{', returns the index of its matching '}',
    respecting string literals and escapes (a brace inside a quoted string
    must never affect depth) — or -1 if the object never closes. Same
    scanning approach as src.rag.chain.find_complete_json_objects, but for
    finding one specific object's end rather than scanning a whole stream.
    """
    depth, in_string, escape = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i
    return -1


def extract_json_object(text: str) -> str:
    """
    Robustly extracts a single JSON object from a model response that may
    have prose before and/or after it — a small model told to "return only
    JSON" (and, for the evidence judge, also told to "show your work") will
    sometimes wrap the object in a ```json fence AND continue with
    unrequested prose afterward, which a simple "strip a fence bracketing
    the whole string" pass (see _strip_code_fences) cannot recover — this
    was found live, not hypothetically, when Stage B's model produced
    exactly this shape mid-eval.

    Strategy, most-specific first:
    1. A fenced ```...``` block anywhere in the text — take the first one's
       contents (handles fence-plus-trailing-prose).
    2. No fence: find the first '{' and its brace-matched '}' (handles bare
       JSON with prose before/after, no fence at all).
    3. Neither found: return the input unchanged (stripped) so the caller's
       own json.loads still runs and fails with a normal error to log.
    """
    text = text.strip()

    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    brace_start = text.find("{")
    if brace_start != -1:
        brace_end = _find_matching_brace(text, brace_start)
        if brace_end != -1:
            return text[brace_start:brace_end + 1]

    return text


def _parse_judge_response(raw: str) -> Dict[str, Any]:
    """
    Never raises. Any malformed or unexpected shape logs the raw response
    and defaults to {"answerable": False, ...} — an unparseable judge
    response must never silently fall through to "answerable", since that
    would defeat the entire point of adding the judge as a gate.
    """
    cleaned = extract_json_object(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("Judge returned malformed JSON, defaulting to answerable=False: %r", raw[:500])
        return {"answerable": False, "reason": "Judge response could not be parsed as JSON."}

    if not isinstance(parsed, dict):
        logger.error("Judge returned non-object JSON, defaulting to answerable=False: %r", raw[:500])
        return {"answerable": False, "reason": "Judge response was not a JSON object."}

    answerable = parsed.get("answerable")
    if not isinstance(answerable, bool):
        logger.error("Judge response missing a boolean 'answerable' field, defaulting to False: %r", raw[:500])
        return {"answerable": False, "reason": "Judge response was missing a valid 'answerable' field."}

    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "No reason given."

    return {"answerable": answerable, "reason": reason.strip()}


def judge_answerability(
    client: anthropic.Anthropic, question: str, chunks: List[RetrievedChunk]
) -> Dict[str, Any]:
    """
    Ask a small model whether `chunks` (the caller passes at most the top
    few retrieved — see chain.JUDGE_CHUNK_LIMIT) contain enough evidence to
    answer `question`, as opposed to merely being topically related to
    diabetes. Returns {"answerable": bool, "reason": str}.

    Never raises: an API failure or a malformed response both default to
    answerable=False rather than crashing the request or, worse, silently
    letting generation proceed unguarded.
    """
    if not chunks:
        return {"answerable": False, "reason": "No excerpts were retrieved for this question."}

    user_content = f"Question: {question}\n\nRetrieved excerpts:\n{_format_chunks_for_judge(chunks)}"
    print_prompt_estimate(f"judge: {question[:60]!r}", JUDGE_SYSTEM_PROMPT, user_content)

    try:
        response = client.messages.create(
            model=get_judge_model(),
            max_tokens=300,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = "".join(block.text for block in response.content if block.type == "text")
    except anthropic.APIError as e:
        logger.error("Judge API call failed, defaulting to answerable=False: %s", e)
        return {"answerable": False, "reason": f"Judge call failed: {e}"}

    result = _parse_judge_response(raw)
    # The exact model string the API actually served the request with —
    # not get_judge_model()'s config value, which could be a request for a
    # model alias that resolves to a different pinned snapshot. Callers
    # that need to report "which model was actually called" (e.g. an eval
    # report's Limitations section) should read this, not the config
    # default, since the two can diverge.
    result["model"] = response.model
    return result
