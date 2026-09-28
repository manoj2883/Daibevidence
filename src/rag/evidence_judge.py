"""
Stage B of the two-stage answerability judge: evidence sufficiency, question + retrieved
chunks. Runs only for questions Stage A (src/rag/scope_judge.py) didn't already reject as
unrelated. Audits the retrieved passages against the question's specific propositions instead
of asking a single "answerable?" boolean — the old boolean judge (src/rag/judge.py) accepted
on topical overlap (ids 1, 10, 22 in the 43-question eval: judged answerable at high
similarity, produced UNSUPPORTED answers), because "does this look related" and "does this
actually state the proposition" collapse into one yes/no in a boolean prompt. This stage keeps
them separate through an explicit decompose -> per-proposition audit -> verdict structure.

EVIDENCE_SYSTEM_PROMPT_TEMPLATE below is the user's revised prompt of 2026-09-28, verbatim apart
from the {question} and passages template variables. It replaced the original task's prompt after
the first two-stage eval; because it was written from failures on the 43-question eval set, results
on that set are development/regression numbers only (see eval/PREDICTION_revised_prompts.md). Do not
edit its wording as part of ordinary maintenance: a prompt change is a deliberate, separate decision,
never a tweak made after looking at eval results.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional

import anthropic

from src.rag.cost import print_prompt_estimate
from src.rag.judge import extract_json_object, get_judge_model
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
   Be concrete: name the intervention, the outcome, and the direction of effect, plus any
   quantity or comparison the question requires. A question asking "how much" needs a
   magnitude. A question asking "which is better" or "X versus Y" needs evidence about
   both X and Y on the same outcome. Mark exactly one proposition as the core: the one
   the question cannot be answered without.

2. ON TOPIC. State whether any passage is about the subject of the question at all. If
   none is, set "on_topic" to false.

3. AUDIT. For each proposition, label each passage that bears on it:
   - DIRECT: the passage states this proposition, in any population
   - PARTIAL: the passage addresses the topic but not this specific claim (a different
     outcome measure, an adjacent intervention, no magnitude where one is required)
   - ABSENT: nothing in the passage bears on it
   Topical similarity is not evidence. A passage about the same disease, nutrient, or
   drug class is ABSENT unless it states the proposition being audited.
   Population does not change the label: a passage that states the proposition in a
   different population is still DIRECT. Population is handled in step 5.

4. COMPARISONS. If a proposition compares X and Y and no single passage compares them,
   it counts as DIRECT only when there are separate DIRECT passages for X and for Y that
   measure the same outcome. If so, set "cross_study_comparison" to true. If the outcome
   measures differ, the proposition is PARTIAL.

5. POPULATION. State the population the question concerns. The default is type 2
   diabetes unless the question says otherwise. List the populations of the DIRECT
   passages for the core proposition. If none of them matches the question's population,
   set "population_mismatch" to true.

6. VERDICT:
   - sufficient: the core proposition has a DIRECT passage in the matching population,
     every other proposition has a DIRECT passage, and "cross_study_comparison" is false
   - partial: the core proposition has a DIRECT passage, but at least one of these is
     true: the population doesn't match, a secondary proposition is PARTIAL or ABSENT,
     or the comparison is assembled across separate studies
   - insufficient: the core proposition has no DIRECT passage in any population

Return only JSON:
{{
  "propositions": [{{"claim": "", "core": true | false, "direct": [], "partial": [], "absent": true | false}}],
  "on_topic": true | false,
  "question_population": "",
  "evidence_populations": [],
  "population_mismatch": true | false,
  "cross_study_comparison": true | false,
  "verdict": "sufficient" | "partial" | "insufficient",
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
        "on_topic": None,
        "cross_study_comparison": False,
        "verdict": "insufficient",
        "reason": reason,
    }


_CHUNK_REF_PATTERN = re.compile(r"^c?(\d+)$", re.IGNORECASE)


def _normalize_chunk_ref(value: Any) -> Optional[int]:
    """
    The prompt labels passages "[C{n}]" — in practice the model sometimes
    returns direct/partial entries as the bare int n, and sometimes as the
    string "C{n}" (or "c{n}") matching the label it was just shown. The
    verbatim prompt (Step 4) doesn't specify which, so both are real,
    legitimate outputs from the same prompt, not a model error — normalize
    both to the plain int every other chunk-index convention in this
    codebase already uses ([Excerpt N], sentence chunk_ids), rather than
    picking one and silently dropping the other's references.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = _CHUNK_REF_PATTERN.match(value.strip())
        if match:
            return int(match.group(1))
    return None


def _normalize_chunk_refs(value: Any) -> List[int]:
    if not isinstance(value, list):
        return []
    return [ref for ref in (_normalize_chunk_ref(v) for v in value) if ref is not None]


def _validate_proposition(p: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(p, dict):
        return None
    claim = p.get("claim")
    if not isinstance(claim, str):
        return None
    direct = _normalize_chunk_refs(p.get("direct"))
    partial = _normalize_chunk_refs(p.get("partial"))
    absent = p.get("absent")
    absent = bool(absent) if isinstance(absent, bool) else (not direct and not partial)
    core = p.get("core")
    if not isinstance(core, bool):
        return None
    return {"claim": claim, "core": core, "direct": direct, "partial": partial, "absent": absent}


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
    # The verdict rules are defined relative to the one core proposition, so a response that
    # marks zero or several cores can't be interpreted: treat it as a schema failure.
    if sum(1 for p in propositions if p["core"]) != 1:
        return None

    on_topic = parsed.get("on_topic")
    cross_study_comparison = parsed.get("cross_study_comparison")
    if not isinstance(on_topic, bool) or not isinstance(cross_study_comparison, bool):
        return None

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
        "on_topic": on_topic,
        "cross_study_comparison": cross_study_comparison,
        "verdict": verdict,
        "reason": reason,
    }


def _call_evidence_model(client: anthropic.Anthropic, question: str, chunks: List[RetrievedChunk]):
    system_prompt = EVIDENCE_SYSTEM_PROMPT_TEMPLATE.format(question=question, passages=_format_passages(chunks))
    print_prompt_estimate(f"evidence: {question[:60]!r}", system_prompt)
    response = client.messages.create(
        model=get_judge_model(),
        # The prompt asks the model to show its work before the JSON, and no extended thinking
        # is enabled, so this budget covers the visible reasoning plus the full JSON. The
        # revised prompt has more steps than the original (which used 1500 with no truncation
        # in the first eval); 4000 leaves ample headroom, and a max_tokens stop is still
        # caught below and treated as a parse failure.
        max_tokens=4000,
        # No temperature param here — see the matching note in
        # src.rag.scope_judge._call_scope_model; this SDK build's
        # Messages.create() doesn't accept one at all.
        system=system_prompt,
        messages=[{"role": "user", "content": "Perform the audit now."}],
    )
    raw = "".join(block.text for block in response.content if block.type == "text")
    if response.stop_reason == "max_tokens":
        logger.error("Evidence judge hit max_tokens before finishing")
        raw = ""  # truncated JSON: force the parse-failure path (retry, then fail closed)
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
        cleaned = extract_json_object(raw)
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
