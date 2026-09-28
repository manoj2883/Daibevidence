"""
Stage B of the two-stage answerability judge: evidence sufficiency, question + retrieved
chunks. Runs only for questions Stage A (src/rag/scope_judge.py) didn't already reject as
unrelated. Audits the retrieved passages against the question's specific propositions instead
of asking a single "answerable?" boolean — the old boolean judge (src/rag/judge.py) accepted
on topical overlap (ids 1, 10, 22 in the 43-question eval: judged answerable at high
similarity, produced UNSUPPORTED answers), because "does this look related" and "does this
actually state the proposition" collapse into one yes/no in a boolean prompt. This stage keeps
them separate through an explicit decompose -> per-proposition audit -> verdict structure.

EVIDENCE_SYSTEM_PROMPT_TEMPLATE below is verbatim from the task that specified this rebuild,
apart from the {question} and passages template variables — do not edit its wording as part
of ordinary maintenance; if the prompt needs to change, that is a deliberate, separate
decision, not a drive-by tweak (see the project's "run the eval exactly once, do not iterate
on judge prompts based on eval results" rule for why: changing this prompt after seeing eval
results would contaminate the evaluation this rebuild is being judged by).
"""
import json
import logging
from typing import Any, Dict, List, Optional

import anthropic

from src.rag.cost import print_prompt_estimate
from src.rag.judge import _strip_code_fences, get_judge_model
from src.rag.types import RetrievedChunk

logger = logging.getLogger("diabevidence")

VALID_VERDICTS = {"sufficient", "partial", "insufficient"}

EVIDENCE_SYSTEM_PROMPT_TEMPLATE = """You decide whether a set of retrieved passages contains enough evidence to answer a
question. You are not answering the question. You are auditing the evidence.

QUESTION: {question}

PASSAGES:
{passages}

Work through these steps in order and show your work.

1. DECOMPOSE. Break the question into the specific factual propositions it asks about.
   Be concrete: name the intervention, the outcome, the population, the direction of
   effect, and any quantity or comparison the question demands. A question asking
   "how much" needs a magnitude; a question asking "which is better" needs a comparison
   between both named things.

2. AUDIT. For each proposition, list every passage id that bears on it and label the
   relationship:
   - DIRECT: the passage states this proposition
   - PARTIAL: the passage addresses the topic but not this specific claim (wrong
     population, wrong outcome measure, adjacent intervention, no magnitude where one
     is required)
   - ABSENT: nothing in the passage bears on it
   Topical similarity is not evidence. A passage about the same disease, nutrient, or
   drug class is ABSENT unless it states the proposition being audited.

3. POPULATION. State the diabetes population the question concerns and the populations
   the DIRECT passages actually studied. Flag any mismatch.

4. VERDICT:
   - sufficient: every proposition has at least one DIRECT passage, populations match
   - partial: the core proposition has DIRECT support but secondary ones are PARTIAL or
     ABSENT, or there is a population mismatch
   - insufficient: the core proposition has no DIRECT support

Return only JSON:
{{
  "propositions": [{{"claim": "", "direct": [], "partial": [], "absent": true|false}}],
  "question_population": "",
  "evidence_populations": [],
  "population_mismatch": true|false,
  "verdict": "sufficient|partial|insufficient",
  "reason": "one sentence, naming the specific gap if not sufficient"
}}"""


def _format_passages(chunks: List[RetrievedChunk]) -> str:
    """
    "[C{n}] population={population_tag} year={year}\\n{chunk_text}" per the task's template —
    deliberately no similarity score in the prompt (Step 4: "Do NOT include similarity scores
    in the prompt"), so the audit can't anchor on retrieval rank/score the way the old boolean
    judge implicitly could by seeing chunks pre-sorted by score.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata or {}
        population = meta.get("population") or "unknown"
        year = meta.get("year") or "unknown"
        blocks.append(f"[C{i}] population={population} year={year}\n{chunk.text}")
    return "\n\n".join(blocks)


def _default_result(reason: str) -> Dict[str, Any]:
    # Fail closed to "insufficient" — never let a parse failure silently
    # pass evidence sufficiency and reach generation ungoverned.
    return {
        "propositions": [],
        "question_population": None,
        "evidence_populations": [],
        "population_mismatch": False,
        "verdict": "insufficient",
        "reason": reason,
    }


def _validate_proposition(p: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(p, dict):
        return None
    claim = p.get("claim")
    if not isinstance(claim, str):
        return None
    direct = p.get("direct")
    partial = p.get("partial")
    absent = p.get("absent")
    direct = direct if isinstance(direct, list) else []
    partial = partial if isinstance(partial, list) else []
    absent = bool(absent) if isinstance(absent, bool) else (not direct and not partial)
    return {"claim": claim, "direct": direct, "partial": partial, "absent": absent}


def _validate_evidence_response(parsed: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(parsed, dict):
        return None
    verdict = parsed.get("verdict")
    if verdict not in VALID_VERDICTS:
        return None
    propositions_raw = parsed.get("propositions")
    if not isinstance(propositions_raw, list):
        return None
    propositions = []
    for p in propositions_raw:
        validated = _validate_proposition(p)
        if validated is None:
            return None
        propositions.append(validated)

    question_population = parsed.get("question_population")
    question_population = question_population if isinstance(question_population, str) else None

    evidence_populations = parsed.get("evidence_populations")
    evidence_populations = [e for e in evidence_populations if isinstance(e, str)] if isinstance(evidence_populations, list) else []

    population_mismatch = parsed.get("population_mismatch")
    population_mismatch = population_mismatch if isinstance(population_mismatch, bool) else False

    reason = parsed.get("reason")
    reason = reason.strip() if isinstance(reason, str) and reason.strip() else "No reason given."

    return {
        "propositions": propositions,
        "question_population": question_population,
        "evidence_populations": evidence_populations,
        "population_mismatch": population_mismatch,
        "verdict": verdict,
        "reason": reason,
    }


def _call_evidence_model(client: anthropic.Anthropic, question: str, chunks: List[RetrievedChunk]):
    system_prompt = EVIDENCE_SYSTEM_PROMPT_TEMPLATE.format(question=question, passages=_format_passages(chunks))
    print_prompt_estimate(f"evidence: {question[:60]!r}", system_prompt)
    response = client.messages.create(
        model=get_judge_model(),
        max_tokens=1500,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": "Perform the audit now."}],
    )
    raw = "".join(block.text for block in response.content if block.type == "text")
    return raw, response.model


def judge_evidence(client: anthropic.Anthropic, question: str, chunks: List[RetrievedChunk]) -> Dict[str, Any]:
    """
    Runs the verbatim Stage B prompt over `chunks` (already capped by the caller) and returns
    the parsed audit dict plus "model". Retries once on a parse/schema failure; fails closed to
    verdict="insufficient" (with the parse error logged) on a second failure or an API error —
    matching Step 4's "parse robustly ... on parse failure retry once, then fail closed to
    insufficient with the parse error logged."

    Never raises. Never given an empty chunk list by design (the caller only invokes Stage B
    after retrieval; a genuinely empty candidate pool is handled by the caller's own defensive
    branch before Stage B would ever be called) — but an empty list here still fails closed
    rather than assuming.
    """
    if not chunks:
        return {**_default_result("No passages were retrieved for this question."), "model": None}

    model_used = None
    last_raw = None
    for attempt in range(2):
        try:
            raw, model_used = _call_evidence_model(client, question, chunks)
        except anthropic.APIError as e:
            logger.error("Evidence judge API call failed (attempt %d): %s", attempt + 1, e)
            break
        last_raw = raw
        cleaned = _strip_code_fences(raw)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error("Evidence judge returned malformed JSON (attempt %d): %r", attempt + 1, raw[:800])
            continue
        validated = _validate_evidence_response(parsed)
        if validated is not None:
            validated["model"] = model_used
            return validated
        logger.error("Evidence judge response failed schema validation (attempt %d): %r", attempt + 1, raw[:800])

    reason = (
        f"Evidence judge failed to return a valid response after retry (last raw: {last_raw[:200]!r})."
        if last_raw
        else "Evidence judge API call failed."
    )
    result = _default_result(reason)
    result["model"] = model_used
    return result
