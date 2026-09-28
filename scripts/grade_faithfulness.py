"""
LLM faithfulness grader: for a set of already-generated answers (identified
by (pipeline, question id) pairs), checks whether each sentence's claim is
actually supported by the excerpts it cites, using a fixed rubric, and
saves per-sentence + overall verdicts to data/eval_v1_faithfulness_v2_5k.json.

"Answered" is not "correct" — src.rag.chain's own SYSTEM_PROMPT rules can
still produce a sentence whose cited excerpt doesn't quite say what the
sentence claims (over-generalizing a specific finding, stretching a
combination-therapy result to a monotherapy question, etc.). This grader
exists to catch that, independently of both the answerability judge and
the main model's own self-reported "supported" flag.

Which (pipeline, id) pairs get graded is decided by
scripts/build_judge_report.py from the two full pipeline runs (never
hand-picked here) — this script just takes that list and grades it.

Usage:
    python -m scripts.grade_faithfulness
"""
import json
import logging
import os

import anthropic
from dotenv import load_dotenv

from src.rag.chain import get_client
from src.rag.judge import _strip_code_fences

load_dotenv()

logger = logging.getLogger("diabevidence.grader")

GRADER_MODEL_DEFAULT = "claude-haiku-4-5"

NEW_PIPELINE_PATH = "data/eval_v1_summary_v2_5k_judge_v2.json"
OLD_PIPELINE_PATH = "data/eval_v1_old_pipeline_full_v2_5k.json"
OUT_PATH = "data/eval_v1_faithfulness_v2_5k.json"

GRADER_SYSTEM_PROMPT = """You are a faithfulness grader for a retrieval-augmented question-answering system. You will \
be given a question, the numbered excerpts that were available to the system, and the system's answer as a numbered \
list of sentences with the excerpt numbers each sentence cites.

For each sentence, grade whether its claim is actually supported by its cited excerpts, using exactly this rubric:
- SUPPORTED: the sentence's claim is directly and specifically stated in at least one cited excerpt.
- PARTIALLY_SUPPORTED: the cited excerpts are relevant and support part of the claim, but the sentence adds a \
specific detail, magnitude, population, or generalization not actually stated in the excerpts, or only loosely \
implies the claim.
- UNSUPPORTED: the cited excerpts do not support the claim, the sentence cites nothing but makes a factual claim, \
or a background-tier excerpt is cited as if it were research evidence.
- NOT_APPLICABLE: pure transition/framing with no factual claim (e.g. "Here is what the evidence shows.").

Then identify which sentence (by number) most directly answers the specific claim in the question — the "central \
claim" — and separately give one overall verdict for the whole answer (SUPPORTED / PARTIALLY_SUPPORTED / \
UNSUPPORTED).

The overall verdict asks "did this answer actually resolve the question," which is a different question from \
"is the central-claim sentence itself factually accurate." Two cases need different handling, and it is important \
not to conflate them:
1. The central claim is a substantive answer to the question, and it is (or isn't) borne out by its excerpts — grade \
this normally: overall verdict follows the central claim's own SUPPORTED/PARTIALLY_SUPPORTED/UNSUPPORTED verdict.
2. The central claim is itself a hedge — a statement that no excerpt addresses the question's specific claim (e.g. \
"no study directly tests X," "no excerpt addresses Y"). Such a hedge can be a perfectly accurate sentence (the \
excerpts genuinely don't address it) and would normally grade SUPPORTED as a claim about the evidence — but that \
accuracy does NOT mean the question was answered. Whenever the central claim is this kind of hedge, the overall \
verdict must be UNSUPPORTED regardless of how accurate the hedge itself is, and regardless of how well-supported \
the other, tangential sentences are. Do not let accurate surrounding context upgrade the overall verdict when the \
question itself was never actually resolved. An answer with no sentence that actually addresses the question's \
central claim (every sentence graded NOT_APPLICABLE or only tangential, or the central claim is a hedge per case 2) \
has central_claim_index: null and overall_verdict "UNSUPPORTED".

Apply this identically across every answer you grade — the same shape (tangential background + an accurate hedge \
that the specific claim isn't addressed) must always get the same overall_verdict "UNSUPPORTED", never SUPPORTED \
just because the hedge sentence itself checks out as accurate.

Respond with ONLY a JSON object and nothing else — no markdown code fences, no explanation outside the object — in \
exactly this shape:
{"sentence_verdicts": [{"index": <int>, "verdict": "<SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED|NOT_APPLICABLE>", \
"note": "<one short sentence>"}, ...], "central_claim_index": <int or null>, "overall_verdict": \
"<SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED>"}"""


def get_grader_model():
    return os.environ.get("ANTHROPIC_GRADER_MODEL", GRADER_MODEL_DEFAULT)


def _format_excerpts(sources):
    blocks = []
    for i, s in enumerate(sources, start=1):
        label = f"PMID {s.get('pmid')}" if s.get("source_type") != "background" else s.get("publisher", "background")
        blocks.append(f"[Excerpt {i}] ({s.get('source_type')}, {label}) {s.get('title', '')}\n{s.get('excerpt', '')}")
    return "\n\n".join(blocks)


def _format_answer(sentence_records):
    lines = []
    for i, s in enumerate(sentence_records, start=1):
        cites = ",".join(str(c) for c in (s.get("chunk_ids") or [])) or "none"
        lines.append(f"{i}. [cites: {cites}] {s.get('sentence', '')}")
    return "\n".join(lines)


def _default_verdict(reason):
    return {"sentence_verdicts": [], "central_claim_index": None, "overall_verdict": "UNSUPPORTED", "grading_error": reason}


def grade_answer(client, question, sources, sentence_records):
    """
    Never raises: a malformed grader response or API failure defaults to
    overall_verdict UNSUPPORTED with grading_error set, rather than
    silently treating a grading failure as a pass.
    """
    if not sentence_records:
        return _default_verdict("no sentences to grade")

    user_content = (
        f"Question: {question}\n\n"
        f"Excerpts available to the system:\n{_format_excerpts(sources)}\n\n"
        f"System's answer:\n{_format_answer(sentence_records)}"
    )

    try:
        response = client.messages.create(
            model=get_grader_model(),
            max_tokens=1500,
            system=GRADER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = "".join(b.text for b in response.content if b.type == "text")
        model_used = response.model
    except anthropic.APIError as e:
        logger.error("Grader API call failed: %s", e)
        return _default_verdict(f"grader call failed: {e}")

    cleaned = _strip_code_fences(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("Grader returned malformed JSON: %r", raw[:500])
        return _default_verdict("malformed grader JSON")

    if not isinstance(parsed, dict) or "overall_verdict" not in parsed:
        logger.error("Grader response missing overall_verdict: %r", raw[:500])
        return _default_verdict("grader response missing overall_verdict")

    overall = parsed.get("overall_verdict")
    if overall not in ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"):
        logger.error("Grader returned invalid overall_verdict %r", overall)
        return _default_verdict(f"invalid overall_verdict: {overall!r}")

    parsed["model"] = model_used
    parsed.setdefault("sentence_verdicts", [])
    parsed.setdefault("central_claim_index", None)
    return parsed


def load_pipeline_results(path):
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in json.load(f)["results"]}


def main():
    load_dotenv()
    with open("data/eval_v1_grading_targets.json", encoding="utf-8") as f:
        targets = json.load(f)["targets"]  # [{"pipeline": "old"|"new", "id": int}, ...]

    new_results = load_pipeline_results(NEW_PIPELINE_PATH)
    old_results = load_pipeline_results(OLD_PIPELINE_PATH)
    client = get_client()

    graded = []
    for t in targets:
        pipeline, qid = t["pipeline"], t["id"]
        record = (new_results if pipeline == "new" else old_results)[qid]
        print(f"  grading pipeline={pipeline} id={qid} {record['question'][:60]!r}")
        verdict = grade_answer(client, record["question"], record.get("sources") or [], record.get("sentence_records") or [])
        graded.append({
            "pipeline": pipeline,
            "id": qid,
            "question": record["question"],
            "final_state": record["final_state"],
            "generated_answer": record.get("generated_answer"),
            "verdict": verdict,
        })
        print(f"    -> overall_verdict={verdict.get('overall_verdict')}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"grader_model_default": get_grader_model(), "results": graded}, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(graded)} gradings to {OUT_PATH}")


if __name__ == "__main__":
    main()
