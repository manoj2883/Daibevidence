"""
Faithfulness grader: checks whether a generated answer's sentences are
actually backed by the source excerpts they cite, independently of the
two-stage answerability judge (src.rag.scope_judge / src.rag.evidence_judge).

Independence, deliberately engineered (Step 8 of the two-stage-judge
rebuild), not assumed:
- Different model. Stage A/B both use src.rag.judge.get_judge_model()
  (Haiku). This grader uses src.rag.chain.get_model() (Sonnet) — a model
  already configured in the repo for generation, reused here rather than
  introducing a third model name, per the task's instruction to prefer an
  already-configured model over a new one.
- Different wording. GRADER_SYSTEM_PROMPT below shares no phrasing with
  EVIDENCE_SYSTEM_PROMPT_TEMPLATE in src/rag/evidence_judge.py — no
  DECOMPOSE/AUDIT/DIRECT/PARTIAL/ABSENT vocabulary, no "topical similarity
  is not evidence" framing. It grades sentence-by-sentence against cited
  excerpts using its own rubric (SUPPORTED/PARTIALLY_SUPPORTED/
  UNSUPPORTED/NOT_APPLICABLE + one overall verdict), arrived at
  independently of how Stage B frames the same underlying problem.
If the two systems still agree most of the time, that agreement means
something; if this grader shared Stage B's model and phrasing, agreement
would be far less informative (correlated blind spots, not independent
confirmation) — see eval/REPORT_twostage_judge.md's limitations section.

Usage:
    python -m scripts.grade_faithfulness data/eval_v2_judge_twostage_v2_5k.json
    python -m scripts.grade_faithfulness data/eval_v2_judge_twostage_v2_5k.json --out data/eval_v2_faithfulness.json

Grades every record whose status is answered/answered_partial/
answered_adjacent (every status that actually generated text) — not a
hand-picked subset. Also writes eval/hand_grading.csv: one row per graded
case with id, question, answer, cited chunk texts, automated grade, and an
empty human_grade column for manual spot-checking.
"""
import argparse
import csv
import json
import logging

import anthropic
from dotenv import load_dotenv

from src.rag.chain import get_client, get_model
from src.rag.judge import extract_json_object

load_dotenv()

logger = logging.getLogger("diabevidence.grader")

ANSWERED_STATUSES = {"answered", "answered_partial", "answered_adjacent"}

GRADER_SYSTEM_PROMPT = """You check whether a written answer's sentences are actually true according to the sources \
it points to. You will see a question, a list of source passages (each with a number), and an answer broken into \
numbered sentences, each sentence showing which passage numbers it points to.

Read each sentence and check it against ONLY the passages it points to (ignore other passages, even if they'd help). \
Assign one of these labels:
- SUPPORTED — the passages it points to actually say this, plainly and specifically.
- PARTIALLY_SUPPORTED — the passages are on-topic and back up part of the sentence, but the sentence adds something \
extra the passages never said (a number, a group of people, a comparison, a stronger or broader claim than the \
passages make).
- UNSUPPORTED — the passages it points to don't say this at all, the sentence points to nothing but still asserts a \
fact, or the sentence leans on a background/informational passage to make a research claim that needs a study.
- NOT_APPLICABLE — the sentence isn't a factual claim at all (a transition, a framing remark).

Then answer one more question, separately from the sentence labels: taking the answer as a whole, did it actually \
resolve what the question was asking? Sometimes every sentence checks out as SUPPORTED individually, and the answer \
still never gets around to the thing the question asked — for instance, three sentences accurately describe related \
background, and a fourth sentence truthfully says "nothing here answers that specific question." Each of those four \
sentences can be perfectly SUPPORTED on its own, but the reader still walks away without an answer. In a case like \
that, your overall judgment should be UNSUPPORTED, because the question itself went unanswered, even though nothing \
in the answer was factually wrong. Give this overall judgment as one of SUPPORTED / PARTIALLY_SUPPORTED / \
UNSUPPORTED, based on whether the question got a real answer, not on whether the individual sentences were accurate.

Reply with ONLY a JSON object, nothing else, no code fences, in exactly this shape:
{"sentence_grades": [{"index": <int>, "label": "<SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED|NOT_APPLICABLE>", \
"why": "<one short sentence>"}, ...], "answers_the_question": <int sentence index that most directly resolves the \
question, or null>, "overall": "<SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED>"}"""


# answered_partial rubric (2026-10-02). A partial answer is designed to answer only what the
# evidence supports and then say what it doesn't cover, so the base rubric's "did it resolve the
# whole question" judgment would mark every honest partial answer down. For partial answers the
# model only labels sentences and reports whether a gap statement is present; the overall grade
# is then computed from those labels in partial_overall() below, not taken from the model.
PARTIAL_RUBRIC_VERSION = "partial_v2"
PARTIAL_RUBRIC = """

THIS ANSWER IS A PARTIAL ANSWER. The system found evidence for only part of the question and is meant to say what the sources do not cover. Grade it on that basis:
- A sentence that says what the sources do not show, cover, or address, or that only describes which people the studies were done in, is NOT_APPLICABLE: it is a statement about the evidence, not a finding.
- Do not penalize the answer for leaving part of the question unanswered.
- Add the key "states_gap": true if at least one sentence tells the reader what the sources do not cover, otherwise false.
Reply with the same JSON object as above, plus "states_gap"."""


def partial_overall(sentence_grades, states_gap):
    """
    SUPPORTED only if every factual sentence (label other than NOT_APPLICABLE) is SUPPORTED by its
    cited passages AND the answer states what is not covered. Missing coverage is not penalized.
    """
    labels = [g.get("label") for g in sentence_grades or [] if isinstance(g, dict)]
    factual = [l for l in labels if l != "NOT_APPLICABLE"]
    if not factual or "UNSUPPORTED" in factual:
        return "UNSUPPORTED"
    if "PARTIALLY_SUPPORTED" in factual or not states_gap:
        return "PARTIALLY_SUPPORTED"
    return "SUPPORTED"


def _format_sources(sources):
    blocks = []
    for i, s in enumerate(sources, start=1):
        label = f"PMID {s.get('pmid')}" if s.get("source_type") != "background" else s.get("publisher", "background")
        blocks.append(f"[{i}] ({s.get('source_type')}, {label}) {s.get('title', '')}\n{s.get('excerpt', '')}")
    return "\n\n".join(blocks)


def _format_answer(sentence_records):
    lines = []
    for i, s in enumerate(sentence_records, start=1):
        cites = ", ".join(str(c) for c in (s.get("chunk_ids") or [])) or "none"
        lines.append(f"{i}. [points to: {cites}] {s.get('sentence', '')}")
    return "\n".join(lines)


def _default_verdict(reason):
    return {"sentence_grades": [], "answers_the_question": None, "overall": "UNSUPPORTED", "grading_error": reason}


def grade_answer(client, question, sources, sentence_records, status=None):
    """
    Never raises: an API failure or malformed response both default to
    overall=UNSUPPORTED with grading_error set, never a silent pass.
    status="answered_partial" grades with PARTIAL_RUBRIC (see partial_overall).
    """
    partial = status == "answered_partial"
    if not sentence_records:
        return _default_verdict("no sentences to grade")

    user_content = (
        f"Question: {question}\n\n"
        f"Source passages:\n{_format_sources(sources)}\n\n"
        f"Answer:\n{_format_answer(sentence_records)}"
    )

    try:
        response = client.messages.create(
            model=get_model(),
            # The grading model emits a thinking block before its text, and
            # thinking tokens count against max_tokens. At 1500, 6 of 10 grades
            # in the first v2 grading pass were truncated or empty (found live;
            # see eval/REPORT_twostage_judge.md). 8000 leaves ample headroom.
            max_tokens=8000,
            system=GRADER_SYSTEM_PROMPT + (PARTIAL_RUBRIC if partial else ""),
            messages=[{"role": "user", "content": user_content}],
        )
        raw = "".join(b.text for b in response.content if b.type == "text")
        model_used = response.model
    except anthropic.APIError as e:
        logger.error("Grader API call failed: %s", e)
        return _default_verdict(f"grader call failed: {e}")

    if response.stop_reason == "max_tokens":
        logger.error("Grader hit max_tokens before finishing (%d output tokens)", response.usage.output_tokens)
        return _default_verdict("grader hit max_tokens")

    cleaned = extract_json_object(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("Grader returned malformed JSON: %r", raw[:500])
        return _default_verdict("malformed grader JSON")

    if not isinstance(parsed, dict) or "overall" not in parsed:
        logger.error("Grader response missing 'overall': %r", raw[:500])
        return _default_verdict("grader response missing 'overall'")

    overall = parsed.get("overall")
    if overall not in ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"):
        logger.error("Grader returned invalid overall %r", overall)
        return _default_verdict(f"invalid overall: {overall!r}")

    parsed["model"] = model_used
    parsed.setdefault("sentence_grades", [])
    if partial:
        if not isinstance(parsed.get("states_gap"), bool):
            logger.error("Partial grade missing boolean 'states_gap': %r", raw[:500])
            return _default_verdict("partial grade missing 'states_gap'")
        parsed["model_overall"] = overall
        parsed["overall"] = partial_overall(parsed["sentence_grades"], parsed["states_gap"])
        parsed["rubric"] = PARTIAL_RUBRIC_VERSION
    parsed.setdefault("answers_the_question", None)
    return parsed


def load_results(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["results"]


def write_hand_grading_csv(path, graded_rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "answer", "cited_chunk_texts", "automated_grade", "human_grade"])
        for row in graded_rows:
            writer.writerow([
                row["id"],
                row["question"],
                row["generated_answer"] or "",
                row["cited_chunk_texts"],
                row["verdict"].get("overall", ""),
                "",
            ])


def _cited_chunk_texts(sources, sentence_records):
    cited_indices = sorted({i for s in sentence_records for i in (s.get("chunk_ids") or [])})
    parts = []
    for i in cited_indices:
        if 1 <= i <= len(sources):
            parts.append(f"[{i}] {sources[i - 1].get('excerpt', '')}")
    return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Grade every answered case in a pipeline results file for faithfulness.")
    parser.add_argument("results_path", help="Path to a pipeline results JSON (must have a top-level 'results' list).")
    parser.add_argument("--out", default=None, help="Output path for verdicts JSON. Defaults to <results_path stem>_faithfulness.json.")
    parser.add_argument("--csv-out", default="eval/hand_grading.csv", help="Output path for the hand-grading CSV.")
    parser.add_argument("--ids", default=None, help="Comma-separated ids to grade; every other graded record is copied from --merge-from.")
    parser.add_argument("--merge-from", default=None, help="Earlier gradings JSON to reuse for ids not in --ids (only records whose answer text is unchanged).")
    args = parser.parse_args()

    out_path = args.out or args.results_path.replace(".json", "_faithfulness.json")

    results = load_results(args.results_path)
    targets = [r for r in results if r.get("status") in ANSWERED_STATUSES]
    print(f"Grading {len(targets)} of {len(results)} records (status in {sorted(ANSWERED_STATUSES)}).")

    reused = {}
    if args.merge_from:
        with open(args.merge_from, encoding="utf-8") as f:
            reused = {g["id"]: g for g in json.load(f)["results"]}
    only = {int(i) for i in args.ids.split(",")} if args.ids else None

    client = get_client()
    graded = []
    for record in targets:
        prior = reused.get(record["id"])
        if only is not None and record["id"] not in only:
            if prior is None or prior.get("generated_answer") != record.get("generated_answer"):
                raise SystemExit(f"id={record['id']} is not in --ids but has no reusable grade for the same answer text.")
            graded.append(prior)
            continue
        print(f"  id={record['id']} status={record['status']} {record['question'][:60]!r}")
        verdict = grade_answer(client, record["question"], record.get("sources") or [], record.get("sentence_records") or [],
                               status=record.get("status"))
        row = {
            "id": record["id"],
            "question": record["question"],
            "status": record["status"],
            "generated_answer": record.get("generated_answer"),
            "cited_chunk_texts": _cited_chunk_texts(record.get("sources") or [], record.get("sentence_records") or []),
            "verdict": verdict,
        }
        graded.append(row)
        print(f"    -> overall={verdict.get('overall')}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"grader_model": get_model(), "source_results_path": args.results_path, "results": graded}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(graded)} gradings to {out_path}")

    write_hand_grading_csv(args.csv_out, graded)
    print(f"Saved hand-grading CSV to {args.csv_out}")


if __name__ == "__main__":
    main()
