"""
v3 eval report: the comparison table, the patient-mode drug-name audit, and the smoke test.
Reads only result files; makes no model calls.

    python -m scripts.report_v3            # prints markdown, writes data/eval_v3_report.md
"""
import json
import os

from src.rag.safety import names_unmentioned_drugs

ANSWERED = {"answered", "answered_partial", "answered_adjacent"}
RUNS = [
    ("Baseline: two-stage judge, revised prompts (v2_5k)", "data/eval_v3_revised_prompts_v2_5k.json",
     "data/eval_v3_revised_prompts_v2_5k_faithfulness.json"),
    ("Run A: v2_5k + background, no scope judge", "data/eval_v3_runA_v2_5k_noscope.json",
     "data/eval_v3_runA_v2_5k_noscope_faithfulness.json"),
    ("Run B: full v3", "data/eval_v3_runB_full.json", "data/eval_v3_runB_full_faithfulness.json"),
]
SMOKE_PATH = "data/eval_v3_smoke.json"
OUT = "data/eval_v3_report.md"


def group(r):
    topic = r.get("topic")
    return "adjacent" if topic == "out_of_scope_diabetes_adjacent" else "unrelated" if topic == "out_of_scope_unrelated" else "in_scope"


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["results"]


def table_row(label, results_path, faith_path):
    if not os.path.exists(results_path):
        return f"| {label} | (not run) | | | |", None
    results = load(results_path)
    counts = {g: [0, 0] for g in ("in_scope", "adjacent", "unrelated")}
    for r in results:
        c = counts[group(r)]
        c[1] += 1
        c[0] += r.get("status") in ANSWERED
    supported = "(not graded)"
    if os.path.exists(faith_path):
        graded = load(faith_path)
        n_sup = sum(1 for g in graded if g["verdict"].get("overall") == "SUPPORTED")
        supported = f"{100 * n_sup / len(graded):.0f}% ({n_sup}/{len(graded)})" if graded else "n/a"
    cell = lambda g: f"{counts[g][0]}/{counts[g][1]}"
    return f"| {label} | {cell('in_scope')} | {cell('adjacent')} | {cell('unrelated')} | {supported} |", results


def drug_audit(results):
    hits = []
    for r in results or []:
        if r.get("audience") != "patient" or r.get("status") not in ANSWERED:
            continue
        sentences = [s["sentence"] for s in r.get("sentence_records") or [] if not s.get("safety_note")]
        found = names_unmentioned_drugs(r["question"], sentences)
        if found:
            hits.append((r["id"], r["question"], found))
    return hits


def main():
    lines = ["| Config | In-scope answered /24 | Adjacent answered /17 | Unrelated /2 | Fully supported % |",
             "|---|---|---|---|---|"]
    run_results = {}
    for label, rp, fp in RUNS:
        row, results = table_row(label, rp, fp)
        lines.append(row)
        run_results[label] = results

    lines += ["", "Patient-mode answers naming a drug/class/supplement the question didn't mention:"]
    for label, _, _ in RUNS[1:]:
        hits = drug_audit(run_results.get(label))
        lines.append(f"- {label}: " + ("none" if not hits else "; ".join(f"id {i} ({', '.join(f)})" for i, _, f in hits)))

    if os.path.exists(SMOKE_PATH):
        lines += ["", "| Smoke question | Status | Sources |", "|---|---|---|"]
        for r in load(SMOKE_PATH):
            srcs = r.get("sources") or r.get("considered_sources") or []
            names = ", ".join(
                f"{s.get('evidence_badge')}: {s.get('publisher') or 'PMID ' + str(s.get('pmid'))}" for s in srcs
            )
            lines.append(f"| {r['question']} | {r['status']} | {names or '-'} |")

    text = "\n".join(lines)
    print(text)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
