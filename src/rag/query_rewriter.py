"""
v3 query step: one Haiku call that replaces the Stage A scope judge.

Returns:
- search_query: the question rewritten into research vocabulary (dense search input)
- keywords: up to 5 search terms, with synonyms and specific names (BM25 input)
- personal_dosing: true if the user asks for a dose or medication change for themselves

INTERNAL ONLY. None of these fields may reach the evidence judge prompt, the
generation prompt, or the user-facing answer: both of those always receive the
user's original question. They are shown only in the retrieval inspector.

Failure handling: on an API or parse failure (after one retry) the step falls
back to the original question, no keywords, personal_dosing=False. Retrieval
still runs on the raw question, and the generation prompt's own medication rule
still forbids dosing advice, so a failed rewrite degrades search quality, not safety.
"""
import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import anthropic

from src.rag.judge import extract_json_object, get_judge_model

logger = logging.getLogger("diabevidence")

MAX_KEYWORDS = 5

QUERY_SYSTEM_PROMPT = """You prepare a user's question for searching a library of diabetes research \
abstracts (PubMed) and patient-education pages. You do not answer the question.

Return only JSON with exactly these keys:
{
  "search_query": "the question rewritten in the vocabulary research abstracts use",
  "keywords": ["up to 5 search terms"],
  "personal_dosing": true | false
}

search_query: keep the question's meaning and population. Replace everyday words with the terms \
researchers use (for example "high sugar in the morning" becomes "morning hyperglycemia", "belly \
fat" becomes "visceral adiposity"). Do not add claims or outcomes the question didn't ask about.

keywords: at most 5 short terms for keyword search. Include synonyms and specific names (drug \
names, diet names, measurements such as HbA1c) that abstracts on this question would contain.

personal_dosing: true only if the person asks what dose, timing, or medication change THEY (or \
someone they care for) should make, e.g. "how much metformin should I take" or "should I stop my \
insulin". General questions about how a medicine works or what studies found are false."""

_cache_lock = threading.Lock()
_cache: Dict[str, Dict[str, Any]] = {}
_disk_cache_path: Optional[Path] = None


def _key(question: str) -> str:
    prompt_hash = hashlib.sha256(QUERY_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]
    normalized = re.sub(r"\s+", " ", (question or "").strip().lower())
    return f"{prompt_hash}|{normalized}"


def enable_disk_cache(path) -> int:
    """Eval scripts persist results so a resumed run never re-pays for a question."""
    global _disk_cache_path
    _disk_cache_path = Path(path)
    if _disk_cache_path.exists():
        _cache.update(json.loads(_disk_cache_path.read_text(encoding="utf-8")))
    return len(_cache)


def _save() -> None:
    if _disk_cache_path is None:
        return
    tmp = _disk_cache_path.with_suffix(_disk_cache_path.suffix + ".tmp")
    tmp.write_text(json.dumps(_cache, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, _disk_cache_path)


def fallback(question: str, reason: str) -> Dict[str, Any]:
    return {"search_query": question, "keywords": [], "personal_dosing": False, "fallback": True, "reason": reason}


def validate(parsed: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(parsed, dict):
        return None
    search_query = parsed.get("search_query")
    keywords = parsed.get("keywords")
    personal_dosing = parsed.get("personal_dosing")
    if not isinstance(search_query, str) or not search_query.strip():
        return None
    if not isinstance(keywords, list) or not isinstance(personal_dosing, bool):
        return None
    clean = []
    for k in keywords:
        if isinstance(k, str) and k.strip() and k.strip().lower() not in {c.lower() for c in clean}:
            clean.append(k.strip())
    return {"search_query": search_query.strip(), "keywords": clean[:MAX_KEYWORDS], "personal_dosing": personal_dosing}


def rewrite_query(client: anthropic.Anthropic, question: str) -> Dict[str, Any]:
    key = _key(question)
    with _cache_lock:
        if key in _cache:
            return {**_cache[key], "cached": True}

    model_used = None
    for attempt in range(2):
        try:
            response = client.messages.create(
                model=get_judge_model(),
                max_tokens=400,
                system=QUERY_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": question}],
            )
        except anthropic.APIError as e:
            logger.error("Query step API call failed (attempt %d): %s", attempt + 1, e)
            return {**fallback(question, f"API call failed: {e}"), "model": None, "cached": False}
        model_used = response.model
        raw = "".join(b.text for b in response.content if b.type == "text")
        try:
            validated = validate(json.loads(extract_json_object(raw)))
        except json.JSONDecodeError:
            validated = None
        if validated is not None:
            validated["model"] = model_used
            with _cache_lock:
                _cache[key] = validated
                _save()
            return {**validated, "fallback": False, "cached": False}
        logger.error("Query step returned invalid JSON (attempt %d): %r", attempt + 1, raw[:300])

    return {**fallback(question, "invalid response after retry"), "model": model_used, "cached": False}
