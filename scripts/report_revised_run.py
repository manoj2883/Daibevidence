"""
Appends a results section for a revised-prompt run to eval/REPORT_twostage_judge.md (never rewrites
the existing report). Everything is computed from the results and faithfulness files.

    python -m scripts.report_revised_run --mode dev
        43-question development/regression run: in-scope rows only, plus answered vs the
        pre-recorded prediction (eval/PREDICTION_revised_prompts.md).
    python -m scripts.report_revised_run --mode heldout --results <file> --questions <file>
        Held-out run: every row, plus answer rate per label group with Wilson 95% intervals.
"""
import argparse
import json

from scripts.build_report_v2 import ANSWERED_STATUSES, eval_label, wilson_ci

REPORT_PATH = "eval/REPORT_twostage_judge.md"
DEV_RESULTS = "data/eval_v3_revised_prompts_v2_5k.json"
PREV_RESULTS = "data/eval_v2_judge_twostage_v2_5k.json"
# Prediction recorded before the run (eval/PREDICTION_revised_prompts.md): the 10 in-scope ids the
# previous two-stage run answered, plus 10, 13, 16, 18 becoming caveated answers.
PREDICTED_FLIPS = {10, 13, 16, 18}
# Ids 9 and 20 were checked against the chunk store before the run (no matching text); the other
# unanswered in-scope ids are corpus gaps only per Stage B's verdict, not a text search.
GREP_CONFIRMED_GAPS = {9, 20}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def faith_by_id(results_path):
    try:
        return {r["id"]: r["verdict"].get("overall") for r in load(results_path.replace(".json", "_faithfulness.json"))["results"]}
    except FileNotFoundError:
        return {}


def row(r, label, faith):
    graded = faith.get(r["id"]) or ("not graded" if r["status"] in ANSWERED_STATUSES else "n/a")
    return f"| {r['id']} | {label} | {r['scope']} | {r['evidence_verdict']} | {r['status']} | {graded} |"


HEADER = ["| id | label | scope | evidence verdict | status | faithfulness |", "|---|---|---|---|---|---|"]


def dev_section():
    labels = {q["id"]: eval_label(q) for q in load("eval/eval_set_v2.json")["questions"]}
    results = {r["id"]: r for r in load(DEV_RESULTS)["results"]}
    prev = {r["id"]: r for r in load(PREV_RESULTS)["results"]}
    faith = faith_by_id(DEV_RESULTS)
    in_scope = sorted(i for i, lab in labels.items() if lab == "in_scope")

    answered = [i for i in in_scope if results[i]["status"] in ANSWERED_STATUSES]
    expected = {i for i in in_scope if prev[i]["status"] in ANSWERED_STATUSES} | PREDICTED_FLIPS
    below = sorted(i for i in expected if i not in answered)
    beyond = sorted(i for i in answered if i not in expected)

    lines = ["", "## 10. Revised prompts: development/regression run on the 43 (2026-09-28)", "",
             "Development/regression numbers only: both revised prompts were written from failures on these 43 questions.", ""]
    lines += HEADER + [row(results[i], "in_scope", faith) for i in in_scope]
    lines += ["", f"In-scope answered: {len(answered)}/24 vs predicted {len(expected)}/24 "
                  f"(ceiling {len(expected)}; {24 - len(expected)} are confirmed corpus gaps).", ""]
    notes = [f"- Corpus-gap caveat: only ids {sorted(GREP_CONFIRMED_GAPS)} were confirmed by searching the chunk store; "
             f"the other {24 - len(expected) - len(GREP_CONFIRMED_GAPS)} are gaps per Stage B's verdict only."]
    for i in below:
        r = results[i]
        reason = r["evidence_reason"] if r["scope"] == "in_charter" else r["scope_reason"]
        notes.append(f"- Below ceiling: id {i} ({r['status']}): {reason}")
    if beyond:
        notes.append(f"- Answered beyond the prediction: ids {beyond}.")
    errors = [r["id"] for r in results.values() if r.get("error")]
    if errors:
        notes.append(f"- Errors: ids {errors}.")
    return lines + notes


def heldout_section(results_path, questions_path):
    questions = load(questions_path)
    questions = questions["questions"] if isinstance(questions, dict) else questions
    labels = {q["id"]: (q.get("label") or eval_label(q)) for q in questions}
    results = {r["id"]: r for r in load(results_path)["results"]}
    faith = faith_by_id(results_path)

    lines = ["", f"## Held-out run: `{questions_path}` (revised prompts)", ""]
    lines += HEADER + [row(results[i], labels[i], faith) for i in sorted(results)]
    lines += ["", "| Label | n | Answered (95% Wilson CI) |", "|---|---|---|"]
    for label in sorted(set(labels.values())):
        ids = [i for i in results if labels[i] == label]
        k = sum(1 for i in ids if results[i]["status"] in ANSWERED_STATUSES)
        p, lo, hi = wilson_ci(k, len(ids))
        lines.append(f"| {label} | {len(ids)} | {k}/{len(ids)} = {p:.1%} ({lo:.1%}–{hi:.1%}) |")
    errors = [r["id"] for r in results.values() if r.get("error")]
    if errors:
        lines += ["", f"- Errors: ids {errors}."]
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dev", "heldout"], required=True)
    parser.add_argument("--results")
    parser.add_argument("--questions")
    args = parser.parse_args()
    lines = dev_section() if args.mode == "dev" else heldout_section(args.results, args.questions)
    text = "\n".join(lines) + "\n"
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
