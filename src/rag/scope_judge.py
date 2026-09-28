"""
Stage A of the two-stage answerability judge: scope only, question only.

Decides whether a question is within DiabEvidence's charter (config/scope_charter.md),
diabetes-adjacent, or unrelated — before retrieval ever runs. This is deliberately blind to
retrieved chunks: a scope decision made after seeing (possibly thin, possibly irrelevant)
retrieval results would let retrieval quality leak into a decision that should depend only on
what the question is about. Evidence sufficiency is Stage B's job (src/rag/evidence_judge.py),
run only for questions Stage A doesn't already reject as unrelated.

config/scope_charter.md is the single source of truth for what counts as in_charter/adjacent/
unrelated — this module loads and embeds its text verbatim, never paraphrases or duplicates it.
"""
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import anthropic

from src.rag.cost import print_prompt_estimate
from src.rag.judge import _strip_code_fences, get_judge_model

logger = logging.getLogger("diabevidence")

CHARTER_PATH = Path(__file__).resolve().parents[2] / "config" / "scope_charter.md"

VALID_SCOPES = {"in_charter", "adjacent", "unrelated"}
VALID_TOPICS = {
    "diet_nutrition",
    "glycemic_control",
    "body_composition_weight",
    "diet_medication_interaction",
}

SCOPE_SYSTEM_PROMPT_TEMPLATE = """You decide whether a question falls within a system's charter, based only on the \
question text — you are never shown any retrieved documents, and your decision must not depend on whether good \
evidence happens to exist. Judge the question's subject matter only.

{charter}

Decide:
- "in_charter": the question asks about one of the four numbered areas above, for adults with type 2 diabetes or \
people caring for them. If so, set "topic" to exactly one of: diet_nutrition, glycemic_control, \
body_composition_weight, diet_medication_interaction.
- "adjacent": the question is clearly about diabetes but outside those four areas (see the Adjacent examples above).
- "unrelated": the question has no connection to diabetes at all.

Respond with ONLY a JSON object and nothing else — no markdown code fences, no explanation outside the object — in \
exactly this shape:
{{"scope": "in_charter" | "adjacent" | "unrelated", "topic": "<one of the four area names, or null>", "reason": \
"<one sentence>"}}"""

_charter_lock = threading.Lock()
_charter_text: Optional[str] = None
_cache_lock = threading.Lock()
_scope_cache: Dict[str, Dict[str, Any]] = {}


def load_charter() -> str:
    """
    Reads config/scope_charter.md once and caches it for the process
    lifetime — the charter is an operator-edited config file, not something
    that changes mid-run, and re-reading it on every call would mean every
    Stage A call pays a filesystem read for no benefit. Restart the process
    to pick up an edit (matches how every other env/config value in this
    project is loaded — see src/ingest/config.py's own module-load-time
    pattern).
    """
    global _charter_text
    if _charter_text is not None:
        return _charter_text
    with _charter_lock:
        if _charter_text is None:
            if not CHARTER_PATH.exists():
                raise FileNotFoundError(
                    f"Scope charter not found at {CHARTER_PATH} — this file is required at startup; "
                    "there is no hardcoded fallback, since the charter is meant to be the single "
                    "editable source of truth for scope."
                )
            _charter_text = CHARTER_PATH.read_text(encoding="utf-8")
    return _charter_text


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", (question or "").strip().lower())


def get_scope_cache() -> Dict[str, Dict[str, Any]]:
    """Exposed for tests/eval scripts that need to clear or inspect the cache between runs."""
    return _scope_cache


def _default_result(reason: str) -> Dict[str, Any]:
    # Fail closed to "unrelated" — the cheapest, safest failure: it refuses
    # before retrieval rather than risking an in_charter/adjacent
    # misjudgment proceeding ungoverned into Stage B and generation.
    return {"scope": "unrelated", "topic": None, "reason": reason}


def _validate_scope_response(parsed: Any) -> Optional[Dict[str, Any]]:
    """Returns a valid, coerced result dict, or None if the shape is wrong (caller retries or fails closed)."""
    if not isinstance(parsed, dict):
        return None
    scope = parsed.get("scope")
    if scope not in VALID_SCOPES:
        return None
    topic = parsed.get("topic")
    if topic is not None and topic not in VALID_TOPICS:
        return None
    if scope != "in_charter":
        topic = None  # topic is only meaningful for in_charter, per the schema
    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "No reason given."
    return {"scope": scope, "topic": topic, "reason": reason.strip()}


def _call_scope_model(client: anthropic.Anthropic, question: str) -> str:
    system_prompt = SCOPE_SYSTEM_PROMPT_TEMPLATE.format(charter=load_charter())
    print_prompt_estimate(f"scope: {question[:60]!r}", system_prompt, question)
    response = client.messages.create(
        model=get_judge_model(),
        max_tokens=300,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    result = "".join(block.text for block in response.content if block.type == "text")
    return result, response.model


def judge_scope(client: anthropic.Anthropic, question: str) -> Dict[str, Any]:
    """
    Returns {"scope": "in_charter"|"adjacent"|"unrelated", "topic": <one of the four
    areas or None>, "reason": str, "model": str, "cached": bool}. Retries once on a parse
    failure (schema mismatch or invalid JSON); on a second failure, or an API error, fails
    closed to "unrelated" with the error logged. Cached in-process, keyed by normalized
    question text (case/whitespace-insensitive) — repeated eval or product-usage runs of the
    same question never pay for a second call.
    """
    key = _normalize_question(question)
    with _cache_lock:
        cached = _scope_cache.get(key)
    if cached is not None:
        return {**cached, "cached": True}

    model_used = None
    last_raw = None
    for attempt in range(2):
        try:
            raw, model_used = _call_scope_model(client, question)
        except anthropic.APIError as e:
            logger.error("Scope judge API call failed (attempt %d): %s", attempt + 1, e)
            break
        last_raw = raw
        cleaned = _strip_code_fences(raw)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error("Scope judge returned malformed JSON (attempt %d): %r", attempt + 1, raw[:500])
            continue
        validated = _validate_scope_response(parsed)
        if validated is not None:
            validated["model"] = model_used
            validated["cached"] = False
            with _cache_lock:
                _scope_cache[key] = {k: v for k, v in validated.items() if k != "cached"}
            return validated
        logger.error("Scope judge response failed schema validation (attempt %d): %r", attempt + 1, raw[:500])

    reason = (
        f"Scope judge failed to return a valid response after retry (last raw: {last_raw[:200]!r})."
        if last_raw
        else "Scope judge API call failed."
    )
    result = _default_result(reason)
    result["model"] = model_used
    result["cached"] = False
    return result
