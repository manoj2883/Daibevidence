"""
Sentence verifier: one Haiku call after the code checks (src.rag.answer_checks), before an answer
is shown. Labels each sentence supported / unsupported / non_factual against the passages that
sentence cites, and says whether it answers the question's core claim, another part, or neither.

decide() then applies the rules, in code:
- unsupported sentences are removed (and logged by the caller); status can only go down:
  answered -> answered_partial when anything is removed;
- no supported sentence answering the core claim survives -> not_covered;
- a partial answer must keep at least one supported sentence answering part of the question,
  otherwise -> not_covered.

The verifier uses the judge model (Haiku). The faithfulness grader (scripts/grade_faithfulness.py)
stays on the generation model (Sonnet), so the two never share a model.

Failure: an API error or two invalid responses fail closed to not_covered. An answer that could
not be verified is not shown.
"""
import json
import logging
from typing import Any, Dict, List, Optional

import anthropic

from src.rag.cost import print_prompt_estimate
from src.rag.judge import extract_json_object, get_judge_model
from src.rag.types import RetrievedChunk

logger = logging.getLogger("diabevidence")

LABELS = {"supported", "unsupported", "non_factual"}
ANSWERS = {"core", "part", "none"}

VERIFIER_PROMPT_TEMPLATE = """You check an answer that was written from research passages, one sentence at a time. \
You do not rewrite anything.

QUESTION: {question}
CORE CLAIM the question needs answered: {core_claim}

PASSAGES:
{passages}

SENTENCES (each shows the passage numbers it cites):
{sentences}

For each sentence give:
- "label":
  - "supported": the passages it cites state what the sentence says. Plain-language rewording is \
fine. Added numbers, populations, comparisons, or a stronger claim than the passages make are not.
  - "unsupported": it makes a factual claim that its cited passages do not state, or it cites \
nothing and still makes a factual claim.
  - "non_factual": it makes no claim about findings: a transition, a statement of what the \
passages do not cover, a description of which people the studies included, or a safety reminder.
- "answers": "core" if the sentence answers the core claim, "part" if it answers another part of \
the question, "none" otherwise.

Judge each sentence only against the passages it cites, never against the others.

Return only JSON: {{"sentences": [{{"index": 1, "label": "supported", "answers": "core"}}]}}"""


def core_claim(question: str, evidence_result: Dict[str, Any]) -> str:
    """The evidence judge's core proposition; the question itself if the judge named none."""
    for p in evidence_result.get("propositions") or []:
        if p.get("core") and p.get("claim"):
            return p["claim"]
    return question


def _format(sentences: List[Dict[str, Any]], chunks: List[RetrievedChunk]) -> Dict[str, str]:
    cited = sorted({c for s in sentences for c in s.get("chunk_ids") or [] if 1 <= c <= len(chunks)})
    passages = "\n\n".join(f"[{c}] {chunks[c - 1].text}" for c in cited) or "(none cited)"
    lines = []
    for i, s in enumerate(sentences, start=1):
        cites = ", ".join(str(c) for c in s.get("chunk_ids") or []) or "none"
        lines.append(f"{i}. [cites: {cites}] {s['sentence']}")
    return {"passages": passages, "sentences": "\n".join(lines)}


def _validate(parsed: Any, n: int) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("sentences"), list):
        return None
    by_index = {}
    for item in parsed["sentences"]:
        if not isinstance(item, dict):
            return None
        idx, label, answers = item.get("index"), item.get("label"), item.get("answers")
        if not isinstance(idx, int) or label not in LABELS or answers not in ANSWERS:
            return None
        by_index[idx] = {"label": label, "answers": answers}
    if sorted(by_index) != list(range(1, n + 1)):
        return None
    return [by_index[i] for i in range(1, n + 1)]


def verify_sentences(client: anthropic.Anthropic, question: str, claim: str,
                     sentences: List[Dict[str, Any]], chunks: List[RetrievedChunk]) -> Dict[str, Any]:
    """Returns {"labels": [{"label", "answers"} per sentence] or None on failure, "model", "error"}."""
    if not sentences:
        return {"labels": [], "model": None, "error": None}
    prompt = VERIFIER_PROMPT_TEMPLATE.format(question=question, core_claim=claim, **_format(sentences, chunks))
    print_prompt_estimate(f"verifier: {question[:60]!r}", prompt)
    model_used, last_raw = None, ""
    for attempt in range(2):
        try:
            response = client.messages.create(
                model=get_judge_model(),
                max_tokens=2000,
                system=prompt,
                messages=[{"role": "user", "content": "Label every sentence now."}],
            )
        except anthropic.APIError as e:
            logger.error("Sentence verifier API call failed: %s", e)
            return {"labels": None, "model": None, "error": f"API call failed: {e}"}
        model_used = response.model
        last_raw = "" if response.stop_reason == "max_tokens" else "".join(
            b.text for b in response.content if b.type == "text")
        try:
            labels = _validate(json.loads(extract_json_object(last_raw)), len(sentences))
        except json.JSONDecodeError:
            labels = None
        if labels is not None:
            return {"labels": labels, "model": model_used, "error": None}
        logger.error("Sentence verifier returned an invalid response (attempt %d): %r", attempt + 1, last_raw[:300])
    return {"labels": None, "model": model_used, "error": f"invalid response after retry: {last_raw[:200]!r}"}


def decide(status: str, sentences: List[Dict[str, Any]], labels: Optional[List[Dict[str, str]]]) -> Dict[str, Any]:
    """
    Pure rules over the verifier's labels. Returns {"outcome": "ok" | "partial" | "not_covered",
    "status": final status or "not_covered", "keep": [indices], "removed": [texts], "reason"}.
    Status only ever goes down.
    """
    if labels is None:
        return {"outcome": "not_covered", "status": "not_covered", "keep": [], "removed": [],
                "reason": "verifier failed; an unverified answer is not shown"}
    keep = [i for i, l in enumerate(labels) if l["label"] != "unsupported"]
    removed = [sentences[i]["sentence"] for i, l in enumerate(labels) if l["label"] == "unsupported"]
    supported = [labels[i] for i in keep if labels[i]["label"] == "supported"]
    new_status = "answered_partial" if removed else status

    if not any(l["answers"] == "core" for l in supported):
        return {"outcome": "not_covered", "status": "not_covered", "keep": [], "removed": removed,
                "reason": "no supported sentence answers the core claim"}
    if new_status == "answered_partial" and not any(l["answers"] in ("core", "part") for l in supported):
        return {"outcome": "not_covered", "status": "not_covered", "keep": [], "removed": removed,
                "reason": "partial answer has no supported sentence answering part of the question"}
    return {"outcome": "partial" if removed else "ok", "status": new_status, "keep": keep, "removed": removed,
            "reason": f"removed {len(removed)} unsupported sentence(s)" if removed else ""}
