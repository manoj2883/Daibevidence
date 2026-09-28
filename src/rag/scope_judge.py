"""
Stage A of the two-stage answerability judge: scope only, question only.

Decides whether a question is within DiabEvidence's charter (config/scope_charter.md),
diabetes-adjacent, or unrelated — before retrieval ever runs. This is deliberately blind to
retrieved chunks: a scope decision made after seeing (possibly thin, possibly irrelevant)
retrieval results would let retrieval quality leak into a decision that should depend only on
what the question is about. Evidence sufficiency is Stage B's job (src/rag/evidence_judge.py),
run only for questions Stage A doesn't already reject as unrelated.

config/scope_charter.md is the single source of truth for what counts as in_charter/adjacent/
unrelated. The prompt's wording (user-written, 2026-09-28) decides by the question's OUTCOME rather
than by the terms it uses; the four-area list and the adjacent examples inside it are read from the
charter's "## Areas" and "## Adjacent examples" sections, never duplicated here.
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
from src.rag.judge import extract_json_object, get_judge_model

logger = logging.getLogger("diabevidence")

CHARTER_PATH = Path(__file__).resolve().parents[2] / "config" / "scope_charter.md"

VALID_SCOPES = {"in_charter", "adjacent", "unrelated"}
VALID_TOPICS = {
    "diet_nutrition",
    "glycemic_control",
    "body_composition_weight",
    "diet_medication_interaction",
}

# User-written prompt (2026-09-28), verbatim apart from {areas} and {adjacent_examples},
# which are read from config/scope_charter.md. Literal JSON braces are doubled for str.format.
SCOPE_SYSTEM_PROMPT_TEMPLATE = """You decide whether a question falls within DiabEvidence's scope. You see only the
question, not any retrieved evidence, and you are not answering it.

DiabEvidence answers questions from adults with type 2 diabetes, or people caring for
them, using published research literature.

Work through these steps in order.

1. OUTCOME. In one short phrase, state what the person ultimately wants to know about:
   the outcome or decision, not the intervention. Examples: "post-meal blood glucose",
   "lean muscle mass", "HbA1c", "whether a food changes how a drug works".

2. POPULATION. If the question is explicitly about type 1 or gestational diabetes, the
   scope is "adjacent". Stop and return.

3. MAP. Match the outcome from step 1 to one of these four areas:
{areas}
   If the outcome maps to one of these areas, the scope is "in_charter", whatever the
   intervention is: a food, a drug, exercise, or surgery.

4. TERMS ARE NOT TOPICS. Words like "insulin", "medication", "drug", "exercise", or
   "surgery" never make a question adjacent on their own. Decide using only the outcome
   from step 1.

5. ADJACENT. Use "adjacent" only when the question is about diabetes but its outcome maps
   to none of the four areas. Examples: {adjacent_examples}.

6. UNRELATED. Use "unrelated" when the question is not about diabetes or metabolic health.

7. PERSONAL DOSING. If the question asks for a dose, schedule, or medication change for a
   specific person, set "personal_dosing" to true. This is a safety flag, separate from
   scope. Classify scope as usual.

8. CHECK. Reread your reason before answering. If the reason names one of the four areas
   as the outcome, the scope must be "in_charter". If you are uncertain between
   "in_charter" and "adjacent", choose "in_charter": a separate evidence check runs next
   and still prevents unsupported answers.

Return only JSON:
{{
  "outcome": "",
  "area": "glycemic_control" | "body_composition_weight" | "diet_nutrition" | "diet_medication_interaction" | null,
  "scope": "in_charter" | "adjacent" | "unrelated",
  "personal_dosing": true | false,
  "reason": "one sentence"
}}"""

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


def _charter_section(charter: str, heading: str) -> str:
    """Body of a "## <heading>" section, up to the next "## " heading. Raises if the section is missing or empty."""
    match = re.search(rf"^## {re.escape(heading)}[ \t]*\n(.*?)(?=^## |\Z)", charter, re.MULTILINE | re.DOTALL)
    body = match.group(1).strip("\n") if match else ""
    if not body.strip():
        raise ValueError(f"config/scope_charter.md has no non-empty '## {heading}' section; Stage A needs it.")
    return body


def build_scope_prompt() -> str:
    """
    Fills the prompt's MAP step with the charter's "## Areas" list (verbatim) and its ADJACENT
    step with the first line of "## Adjacent examples". Fails loudly if any of the four area keys
    is missing from the charter's Areas section, since the validator checks "area" against them.
    """
    charter = load_charter()
    areas = _charter_section(charter, "Areas").rstrip()
    missing = sorted(t for t in VALID_TOPICS if f"- {t}:" not in areas)
    if missing:
        raise ValueError(f"config/scope_charter.md '## Areas' is missing area key(s): {missing}")
    adjacent_examples = _charter_section(charter, "Adjacent examples").strip().splitlines()[0].strip().rstrip(".")
    return SCOPE_SYSTEM_PROMPT_TEMPLATE.format(areas=areas, adjacent_examples=adjacent_examples)


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", (question or "").strip().lower())


def get_scope_cache() -> Dict[str, Dict[str, Any]]:
    """Exposed for tests/eval scripts that need to clear or inspect the cache between runs."""
    return _scope_cache


def _default_result(reason: str) -> Dict[str, Any]:
    # Fail closed to "unrelated" — the cheapest, safest failure: it refuses
    # before retrieval rather than risking an in_charter/adjacent
    # misjudgment proceeding ungoverned into Stage B and generation.
    return {"scope": "unrelated", "area": None, "topic": None, "outcome": None, "personal_dosing": False, "reason": reason}


def _validate_scope_response(parsed: Any) -> Optional[Dict[str, Any]]:
    """
    Returns a valid, coerced result dict, or None if the shape is wrong (caller retries or fails
    closed). "topic" is kept as an alias of "area" so existing consumers (API, eval runner, report
    builder) keep working. personal_dosing must be a real bool: it drives a safety banner, so a
    missing or malformed flag is a schema failure rather than a silent False.
    """
    if not isinstance(parsed, dict):
        return None
    scope = parsed.get("scope")
    if scope not in VALID_SCOPES:
        return None
    area = parsed.get("area", parsed.get("topic"))
    if area is not None and area not in VALID_TOPICS:
        return None
    if scope != "in_charter":
        area = None  # area is only meaningful for in_charter
    personal_dosing = parsed.get("personal_dosing")
    if not isinstance(personal_dosing, bool):
        return None
    outcome = parsed.get("outcome")
    outcome = outcome.strip() if isinstance(outcome, str) and outcome.strip() else None
    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "No reason given."
    return {
        "scope": scope,
        "area": area,
        "topic": area,
        "outcome": outcome,
        "personal_dosing": personal_dosing,
        "reason": reason.strip(),
    }


def _call_scope_model(client: anthropic.Anthropic, question: str) -> str:
    system_prompt = build_scope_prompt()
    print_prompt_estimate(f"scope: {question[:60]!r}", system_prompt, question)
    response = client.messages.create(
        model=get_judge_model(),
        # No extended thinking is enabled for the judge model, so this budget covers only the
        # JSON (outcome + area + reason). 600 leaves headroom over the ~150 tokens it needs;
        # a max_tokens stop is still caught below and treated as a parse failure.
        max_tokens=600,
        # Step 3 asks for temperature 0; the installed anthropic SDK's
        # Messages.create() has no temperature parameter at all in this
        # environment (verified against the live signature, not assumed —
        # it raises TypeError: unexpected keyword argument 'temperature'),
        # so determinism can't be forced here. Caching (see judge_scope's
        # in-process cache) still makes a repeated identical question
        # deterministic within a process; a genuinely new call to the
        # model is not. Noted as a limitation, not silently dropped.
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    result = "".join(block.text for block in response.content if block.type == "text")
    if response.stop_reason == "max_tokens":
        logger.error("Scope judge hit max_tokens before finishing")
        result = ""  # truncated JSON: force the parse-failure path (retry, then fail closed)
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
        cleaned = extract_json_object(raw)
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
