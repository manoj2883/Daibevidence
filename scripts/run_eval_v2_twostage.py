"""
Runs all 43 questions in eval/eval_set_v2.json through the two-stage
answerability judge (src.rag.chain.stream_answer, production defaults — no
retrieval_mode/force_judge diagnostic overrides) exactly once, and saves
full per-question detail to data/eval_v2_judge_twostage_v2_5k.json.

Never overwrites an existing output file (per the task's "run the eval
exactly once" rule) — pass --force only to intentionally replace a run that
was itself a mistake (e.g. an ingestion bug caught before any analysis of
the results), never to re-run after seeing a result you don't like.

Does not modify eval/eval_set_v2.json or its labels.

Usage:
    python -m scripts.run_eval_v2_twostage --namespace v2_5k
"""
import argparse
import json
import os
import time

from dotenv import load_dotenv

from src.rag.chain import stream_answer

QUESTIONS_PATH = "eval/eval_set_v2.json"
OUT_PATH = "data/eval_v2_judge_twostage_v2_5k.json"
# Fields added by the revised judge prompts (2026-09-28); absent (None) in runs before that.
JUDGE_EXTRA_FIELDS = ("outcome", "area", "personal_dosing", "on_topic", "cross_study_comparison")


def load_questions(path=QUESTIONS_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["questions"]


def run_one_question(q, namespace):
    t0 = time.monotonic()
    sources = []
    sentence_records = []
    contradictions = []
    status = None
    scope = None
    scope_reason = None
    evidence_verdict = None
    evidence_reason = None
    propositions = []
    population_mismatch = False
    question_population = None
    evidence_populations = []
    top_score = None
    score_distribution = None
    timing_ms = {}
    usage = None
    truncated = False
    model_generation = None
    model_scope = None
    model_evidence = None
    error = None
    refusal_message = None
    judge_extra = {k: None for k in JUDGE_EXTRA_FIELDS}

    for ev in stream_answer(q["question"], namespace=namespace):
        event, data = ev["event"], ev["data"]
        if event in ("sources", "refusal"):
            status = data.get("status")
            scope = data.get("scope")
            scope_reason = data.get("scope_reason")
            evidence_verdict = data.get("evidence_verdict")
            evidence_reason = data.get("evidence_reason")
            propositions = data.get("propositions") or []
            population_mismatch = data.get("population_mismatch", False)
            question_population = data.get("question_population")
            evidence_populations = data.get("evidence_populations") or []
            score_distribution = data.get("score_distribution")
            top_score = (score_distribution or {}).get("top_score")
            timing_ms.update(data.get("timing_ms") or {})
            model_scope = data.get("scope_model")
            judge_extra = {k: data.get(k) for k in JUDGE_EXTRA_FIELDS}
            judge = data.get("judge") or {}
            if judge.get("verdict") is not None:
                model_evidence = judge.get("model")
            if event == "sources":
                sources = data.get("sources") or []
            else:
                refusal_message = data.get("message")
        elif event == "contradictions":
            contradictions = data.get("contradictions") or []
        elif event == "sentence":
            sentence_records.append(data)
        elif event == "done":
            status = data.get("status", status)
            timing_ms.update(data.get("timing_ms") or {})
            usage = data.get("usage")
            truncated = data.get("truncated", False)
            model_generation = data.get("model")
        elif event == "error":
            error = data.get("message")

    wall_ms = round((time.monotonic() - t0) * 1000, 1)
    generated_answer = " ".join(s["sentence"] for s in sentence_records) if sentence_records else None

    return {
        "id": q["id"],
        "topic": q["topic"],
        "in_scope": q["in_scope"],
        "question": q["question"],
        "reference_answer": q.get("reference_answer"),
        "status": status,
        "scope": scope,
        "scope_reason": scope_reason,
        "evidence_verdict": evidence_verdict,
        "evidence_reason": evidence_reason,
        "propositions": propositions,
        "population_mismatch": population_mismatch,
        "question_population": question_population,
        "evidence_populations": evidence_populations,
        **judge_extra,
        "top_score": top_score,
        "score_distribution": score_distribution,
        "generated_answer": generated_answer,
        "refusal_message": refusal_message,
        "sources": sources,
        "sentence_records": sentence_records,
        "num_contradictions": len(contradictions),
        "error": error,
        "timing_ms": timing_ms,
        "wall_ms": wall_ms,
        "usage": usage,
        "truncated": truncated,
        "model_generation": model_generation,
        "model_scope": model_scope,
        "model_evidence": model_evidence,
    }


def main():
    parser = argparse.ArgumentParser(description="Run all 43 questions through the two-stage judge exactly once.")
    parser.add_argument("--namespace", required=True, help="Pinecone namespace to evaluate against, e.g. v2_5k.")
    parser.add_argument("--out", default=OUT_PATH, help="Output path.")
    parser.add_argument("--label", default="two-stage judge (scope + evidence audit)", help="Pipeline label stored in the output file.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file. Use only to replace a run invalidated by a code bug, never after seeing results you don't like.")
    args = parser.parse_args()

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(
            f"{args.out} already exists. This eval is meant to run exactly once; pass --force only if the "
            "existing file is known-invalid (e.g. a code bug caught before any analysis), never to re-run "
            "after seeing results you don't like."
        )

    load_dotenv()
    questions = load_questions()
    print(f"Loaded {len(questions)} questions. Namespace: {args.namespace!r}")

    results = []
    for q in questions:
        print(f"  [{q['topic']:30s}] id={q['id']} {q['question'][:60]!r}")
        record = run_one_question(q, namespace=args.namespace)
        results.append(record)
        print(f"    -> status={record['status']} scope={record['scope']} evidence_verdict={record['evidence_verdict']}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"pipeline": args.label, "namespace": args.namespace, "results": results}, f, ensure_ascii=False, indent=2)

    status_counts = {}
    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    print(f"\nSaved {len(results)} results to {args.out}")
    print("Status counts:", status_counts)
    print("Errors:", sum(1 for r in results if r["error"]))


if __name__ == "__main__":
    main()
