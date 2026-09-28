"""
Reconstructs the OLD (pre-judge) pipeline end-to-end for all 43 questions
in eval/eval_set_v2.json: the similarity floor decides which chunks survive
per tier (retrieval_mode="floor" on src.rag.chain.stream_answer), and the
judge is bypassed entirely (force_judge=True) so anything that clears the
floor goes straight to real Sonnet generation — exactly what the system did
before the answerability judge existed. A question whose floor-survivors
are empty in every tier still short-circuits to a zero-cost refusal via
stream_answer's own defensive out_of_scope branch, unchanged.

This exists because the previous eval report's old-vs-new comparison mixed
two different measurements: the "old" out-of-scope number came from a
retrieval-only run (no generation), while the "old" in-scope number came
from a partial 11-question judge-bypass run. Both sides of the comparison
now come from the same full, real, end-to-end run — see
scripts/build_judge_report.py, which reads this file and
data/eval_v1_summary_v2_5k_judge_v2.json (the new pipeline's equivalent
full run) and computes every reported number from the two.

Usage:
    python -m scripts.run_old_pipeline_full --namespace v2_5k
"""
import argparse
import json

from dotenv import load_dotenv

from scripts.run_eval_v1 import load_questions, run_one_question

# Deliberately NOT scripts.run_eval_v1.QUESTIONS_PATH — that constant
# points at the older 30-question eval_set_v1.json. This script exists
# specifically to reconstruct the old pipeline against the 43-question
# eval_set_v2.json that the judge comparison is about; defaulting to v1
# silently produced a 30-result file with mismatched ids the first time
# this ran (caught by build_judge_report.py's own consistency assertions,
# which is exactly why they exist).
QUESTIONS_PATH = "eval/eval_set_v2.json"
OUT_PATH = "data/eval_v1_old_pipeline_full_v2_5k.json"


def main():
    parser = argparse.ArgumentParser(description="Run all 43 questions through the old (pre-judge) pipeline end-to-end.")
    parser.add_argument("--namespace", required=True, help="Pinecone namespace to evaluate against, e.g. v2_5k.")
    parser.add_argument("--questions-path", default=QUESTIONS_PATH, help="Path to the question-set JSON.")
    parser.add_argument("--out", default=OUT_PATH, help="Output path for the full result set.")
    args = parser.parse_args()

    load_dotenv()

    questions = load_questions(args.questions_path)
    print(f"Loaded {len(questions)} questions. Namespace: {args.namespace!r}")
    print("Pipeline: similarity floor gate (retrieval_mode='floor'), judge bypassed (force_judge=True), real Sonnet generation.")

    results = []
    for q in questions:
        print(f"  [{q['topic']:30s}] id={q['id']} {q['question'][:60]!r}")
        record = run_one_question(q, namespace=args.namespace, retrieval_mode="floor", force_judge=True)
        record["pipeline"] = "old_floor_gate_judge_bypassed"
        results.append(record)
        print(f"    -> final_state={record['final_state']}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pipeline": "old (similarity-floor gate, no judge, real Sonnet generation on floor survivors)",
                "namespace": args.namespace,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    n_out_of_scope_gate = sum(1 for r in results if r["final_state"] == "out_of_scope")
    n_generated = sum(1 for r in results if r["final_state"] not in ("out_of_scope", None) and r["error"] is None)
    n_errors = sum(1 for r in results if r["error"])
    print(f"\nSaved {len(results)} results to {args.out}.")
    print(f"Floor-gate refusals (zero generation cost): {n_out_of_scope_gate}")
    print(f"Generation calls made: {n_generated}")
    print(f"Errors: {n_errors}")


if __name__ == "__main__":
    main()
