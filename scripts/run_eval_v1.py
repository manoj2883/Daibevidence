"""
Runs eval/eval_set_v1.json (30 questions: 24 in-scope, evenly split across
the four ingestion topics, + 6 out-of-scope) through the real, live
pipeline (src.rag.chain.stream_answer) against a given Pinecone namespace.

Distinct from scripts/run_evaluation.py, which runs the older
eval/test_questions.json + eval/out_of_scope_questions.json pair (different
schema: category/target_population/expected_behavior rubric, no ground-truth
reference answers). This script's question set has topic/in_scope/
reference_answer instead, so results are reported per-topic and each
record carries the RAG-generated answer text next to the hand-written
ground truth for manual comparison — reference_answer is never used to
grade or bias generation, only surfaced alongside the output.

Usage:
    python -m scripts.run_eval_v1 --namespace v2_5k --out v2_5k
    python -m scripts.run_eval_v1 --namespace v2_5k --out v2_5k --resume
    python -m scripts.run_eval_v1 --namespace v2_5k --out v2_5k --dry-run
"""
import argparse
import json
import os
import statistics
import time

from dotenv import load_dotenv

from src.ingest.population import classify_population
from src.rag.chain import (
    is_population_mismatch,
    retrieved_population_set,
    stream_answer,
)
from src.rag.reranker import rerank_scores
from src.rag.retriever import (
    decide_retrieval_state,
    get_candidate_k,
    get_similarity_floor,
    get_similarity_floor_background,
    retrieve,
    retrieve_with_floor,
)

QUESTIONS_PATH = "eval/eval_set_v1.json"


def load_questions(path=QUESTIONS_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["questions"]


def run_one_question_retrieval_only(q, namespace=None, rerank=False):
    """
    Pinecone + local-embeddings only — zero Claude calls. Reports the
    pre-generation state guess and retrieved sources, but cannot produce
    a generated answer, per-sentence groundedness, or contradiction
    detection: those only exist after Claude writes the cited answer.

    rerank=True additionally scores the same candidate pool with a local
    cross-encoder (src.rag.reranker, fastembed ONNX, no API cost) to test
    whether a query-passage joint scorer separates in-scope from
    diabetes-adjacent-off-topic content better than cosine similarity
    alone (see data/eval_v1_retrieval_only_v2_5k.json's finding that it
    cannot). Retrieves the full candidate_k pool directly (not just the
    floor survivors) so every candidate's text is available to rerank.
    """
    t0 = time.monotonic()
    requested_population = classify_population(q["question"])
    candidates = retrieve(q["question"], k=get_candidate_k(), namespace=namespace)
    decision = decide_retrieval_state(
        candidates, get_similarity_floor(), background_floor=get_similarity_floor_background()
    )
    wall_ms = round((time.monotonic() - t0) * 1000, 1)

    retrieved = retrieved_population_set(decision.surviving_chunks)
    mismatch = is_population_mismatch(requested_population, retrieved)

    sources = []
    for c in decision.surviving_chunks:
        meta = c.metadata or {}
        sources.append({
            "source_type": meta.get("source_type", "evidence"),
            "pmid": meta.get("pmid", ""),
            "title": meta.get("title", ""),
            "population": meta.get("population", ""),
            "score": c.score,
        })

    result = {
        "id": q["id"],
        "topic": q["topic"],
        "in_scope": q["in_scope"],
        "question": q["question"],
        "reference_answer": q.get("reference_answer"),
        "generated_answer": None,
        "provisional_state": decision.state,
        "requested_population": sorted(requested_population),
        "retrieved_populations": sorted(retrieved),
        "population_mismatch": mismatch,
        "num_surviving_chunks": len(decision.surviving_chunks),
        "top_score": decision.top_score,
        "candidate_scores": decision.candidate_scores,
        "sources": sources,
        "wall_ms": wall_ms,
    }

    if rerank:
        rerank_ms_t0 = time.monotonic()
        ce_scores = rerank_scores(q["question"], [c.text for c in candidates])
        result["rerank_scores"] = sorted(ce_scores, reverse=True)
        result["rerank_ms"] = round((time.monotonic() - rerank_ms_t0) * 1000, 1)

    return result


def summarize_retrieval_only(results):
    in_scope = [r for r in results if r["in_scope"]]
    out_scope = [r for r in results if not r["in_scope"]]

    def rate(items, pred):
        return round(sum(1 for i in items if pred(i)) / len(items), 4) if items else None

    topics = sorted({r["topic"] for r in in_scope})
    per_topic = {}
    for topic in topics:
        items = [r for r in in_scope if r["topic"] == topic]
        per_topic[topic] = {
            "n": len(items),
            "provisional_answered": sum(1 for r in items if r["provisional_state"] == "answered"),
            "provisional_answered_low_confidence": sum(1 for r in items if r["provisional_state"] == "answered_low_confidence"),
            "provisional_out_of_scope": sum(1 for r in items if r["provisional_state"] == "out_of_scope"),
            "population_mismatch": sum(1 for r in items if r["population_mismatch"]),
        }

    return {
        "mode": "retrieval_only (no Claude calls — provisional pre-generation state only)",
        "n_in_scope": len(in_scope),
        "n_out_of_scope": len(out_scope),
        "per_topic": per_topic,
        "out_of_scope_provisional_refusal_rate": rate(out_scope, lambda r: r["provisional_state"] == "out_of_scope"),
        "in_scope_provisional_would_answer_rate": rate(in_scope, lambda r: r["provisional_state"] in ("answered", "answered_low_confidence")),
        "in_scope_provisional_out_of_scope_rate": rate(in_scope, lambda r: r["provisional_state"] == "out_of_scope"),
    }


def run_one_question(q, namespace=None, **stream_kwargs):
    """
    Runs the live pipeline (src.rag.chain.stream_answer). By default this
    is the production judge-gated pipeline; passing retrieval_mode="floor"
    and/or force_judge=<bool> (both diagnostic-only kwargs on
    stream_answer — see its docstring) reconstructs the pre-judge pipeline
    instead, e.g. for scripts/run_old_pipeline_full.py.

    Captures full detail — not just aggregate counts — because
    scripts/grade_faithfulness.py needs the actual cited excerpt text and
    per-sentence chunk_ids to check citations, not just "how many sentences
    were supported": "sources" (full source payloads, excerpt text
    included), "sentence_records" (full per-sentence {sentence, chunk_ids,
    pmids, supported, source_type} objects), "judge" (the full verdict
    dict, including "bypassed": True when force_judge was used — never
    silently indistinguishable from a real verdict), and "model_generation"
    /"model_judge" (the exact model strings the API actually served the
    requests with, read from the response objects — see
    src.rag.chain/src.rag.judge — not from config defaults, which a report
    should never assume match what was actually called).
    """
    t0 = time.monotonic()
    sources = []
    sentence_records = []
    contradictions = []
    final_state = None
    timing_ms = {}
    usage = None
    truncated = False
    error = None
    refusal_message = None
    top_score = None
    judge = None
    score_distribution = None
    model_generation = None

    for ev in stream_answer(q["question"], namespace=namespace, **stream_kwargs):
        event, data = ev["event"], ev["data"]
        if event == "sources":
            sources = data.get("sources", [])
            score_distribution = data.get("score_distribution")
            top_score = (score_distribution or {}).get("top_score")
            judge = data.get("judge")
        elif event == "contradictions":
            contradictions = data.get("contradictions", [])
        elif event == "sentence":
            sentence_records.append(data)
        elif event == "done":
            final_state = data.get("state")
            timing_ms = data.get("timing_ms", {})
            usage = data.get("usage")
            truncated = data.get("truncated", False)
            model_generation = data.get("model")
        elif event == "refusal":
            final_state = data.get("state", "out_of_scope")
            timing_ms = data.get("timing_ms", {})
            refusal_message = data.get("message")
            score_distribution = data.get("score_distribution")
            top_score = (score_distribution or {}).get("top_score")
            judge = data.get("judge")
        elif event == "error":
            error = data.get("message")

    wall_ms = round((time.monotonic() - t0) * 1000, 1)

    evidence_supported = sum(1 for s in sentence_records if s.get("source_type") == "evidence")
    background_supported = sum(1 for s in sentence_records if s.get("source_type") == "background")
    unsupported = sum(1 for s in sentence_records if not s.get("chunk_ids"))
    generated_answer = " ".join(s["sentence"] for s in sentence_records) if sentence_records else None

    return {
        "id": q["id"],
        "topic": q["topic"],
        "in_scope": q["in_scope"],
        "question": q["question"],
        "reference_answer": q.get("reference_answer"),
        "generated_answer": generated_answer,
        "final_state": final_state,
        "top_score": top_score,
        "judge_verdict": (judge or {}).get("answerable"),
        "judge_reason": (judge or {}).get("reason"),
        "judge": judge,
        "score_distribution": score_distribution,
        "answered": final_state is not None and final_state != "out_of_scope" and error is None,
        "error": error,
        "refusal_message": refusal_message,
        "sources": sources,
        "sentence_records": sentence_records,
        "num_sources": len(sources),
        "pmids_cited": sorted({p for s in sentence_records for p in s.get("pmids", []) if p}),
        "num_contradictions": len(contradictions),
        "total_sentences": len(sentence_records),
        "evidence_supported_sentences": evidence_supported,
        "background_supported_sentences": background_supported,
        "unsupported_sentences": unsupported,
        "timing_ms": timing_ms,
        "wall_ms": wall_ms,
        "usage": usage,
        "truncated": truncated,
        "model_generation": model_generation,
        "model_judge": (judge or {}).get("model"),
    }


def load_checkpoint(path):
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[rec["id"]] = rec
    return done


def load_old_baseline(path):
    """
    The pre-judge similarity-floor baseline: a cached retrieval-only run's
    per-question "provisional_state" (see run_one_question_retrieval_only),
    keyed by question id. Used by summarize() to compare the old
    threshold-only gate against the new judge gate on the same 43
    questions, without re-running retrieval. Returns {} (comparison
    skipped) if the file doesn't exist rather than failing the whole eval
    run over a missing optional baseline.
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {r["id"]: r["provisional_state"] for r in data.get("results", [])}


def _gate_breakdown(items, refused_pred, refused_label, answered_label):
    """
    Confusion-style breakdown of one gate's refuse/answer decision against
    ground-truth in_scope, for one set of results that all share the same
    in_scope value (see callers). refused_pred(item) -> bool decides
    whether that gate refused this question. refused_label/answered_label
    name what "refused"/"answered" actually means for this bucket — e.g.
    for out-of-scope questions, "refused" is a correct refusal and
    "answered" is a false accept; for in-scope questions it's the reverse
    (refused = false reject, answered = correct answer) — spelling that out
    in the field names avoids having to remember which bucket is which.
    """
    return {
        "n": len(items),
        refused_label: sum(1 for r in items if refused_pred(r)),
        answered_label: sum(1 for r in items if not refused_pred(r)),
        f"ids_{refused_label}": sorted(r["id"] for r in items if refused_pred(r)),
        f"ids_{answered_label}": sorted(r["id"] for r in items if not refused_pred(r)),
    }


def summarize(results, old_baseline=None):
    old_baseline = old_baseline or {}
    in_scope = [r for r in results if r["in_scope"]]
    out_scope = [r for r in results if not r["in_scope"]]

    def rate(items, pred):
        return round(sum(1 for i in items if pred(i)) / len(items), 4) if items else None

    def mean(values):
        values = [v for v in values if v is not None]
        return round(statistics.mean(values), 2) if values else None

    generated = [r for r in results if r["final_state"] not in ("out_of_scope", None) and r["error"] is None]
    total_sentences_all = sum(r["total_sentences"] for r in generated)
    evidence_sentences_all = sum(r["evidence_supported_sentences"] for r in generated)
    background_sentences_all = sum(r["background_supported_sentences"] for r in generated)

    topics = sorted({r["topic"] for r in in_scope})
    per_topic = {}
    for topic in topics:
        items = [r for r in in_scope if r["topic"] == topic]
        per_topic[topic] = {
            "n": len(items),
            "answered": sum(1 for r in items if r["final_state"] == "answered"),
            "answered_low_confidence": sum(1 for r in items if r["final_state"] == "answered_low_confidence"),
            "no_evidence_for_claim": sum(1 for r in items if r["final_state"] == "no_evidence_for_claim"),
            "out_of_scope": sum(1 for r in items if r["final_state"] == "out_of_scope"),
            "errors": sum(1 for r in items if r["error"]),
        }

    summary = {
        "n_in_scope": len(in_scope),
        "n_out_of_scope": len(out_scope),
        "per_topic": per_topic,
        "out_of_scope_refusal_rate": rate(out_scope, lambda r: r["final_state"] == "out_of_scope"),
        "in_scope_answer_rate": rate(in_scope, lambda r: r["final_state"] in ("answered", "answered_low_confidence")),
        "no_evidence_for_claim_rate_in_scope": rate(in_scope, lambda r: r["final_state"] == "no_evidence_for_claim"),
        "answered_low_confidence_rate_in_scope": rate(in_scope, lambda r: r["final_state"] == "answered_low_confidence"),
        "groundedness_overall": round((evidence_sentences_all + background_sentences_all) / total_sentences_all, 4) if total_sentences_all else None,
        "groundedness_evidence_only": round(evidence_sentences_all / total_sentences_all, 4) if total_sentences_all else None,
        "groundedness_background_only": round(background_sentences_all / total_sentences_all, 4) if total_sentences_all else None,
        "total_sentences": total_sentences_all,
        "total_contradictions_detected": sum(r["num_contradictions"] for r in results),
        "truncated_responses": sum(1 for r in results if r.get("truncated")),
        "mean_latency_ms_out_of_scope": mean([r["timing_ms"].get("retrieval") for r in out_scope]),
        "mean_latency_ms_generated_total": mean([r["timing_ms"].get("total") for r in generated]),
        "mean_input_tokens": mean([r["usage"]["input_tokens"] for r in generated if r["usage"]]),
        "mean_output_tokens": mean([r["usage"]["output_tokens"] for r in generated if r["usage"]]),
        "errors": sum(1 for r in results if r["error"]),
    }

    if old_baseline:
        # Old gate = pure similarity floor (cached provisional_state from a
        # retrieval-only run, e.g. data/eval_v1_retrieval_only_rerank_v2_5k.json).
        # New gate = this run's actual outcome (judge + generation).
        # "Correct refusal" / "false reject" / "false accept" are always
        # relative to the ground-truth in_scope label, never to each other.
        old_refused = lambda r: old_baseline.get(r["id"]) == "out_of_scope"  # noqa: E731
        new_refused = lambda r: r["final_state"] == "out_of_scope" or not r["answered"]  # noqa: E731

        summary["old_vs_new_gate_comparison"] = {
            "old_gate": "similarity floor only (cached retrieval-only provisional_state)",
            "new_gate": "answerability judge + generation (this run)",
            "old_out_of_scope": _gate_breakdown(out_scope, old_refused, "correct_refusals", "false_accepts"),
            "new_out_of_scope": _gate_breakdown(out_scope, new_refused, "correct_refusals", "false_accepts"),
            "old_in_scope": _gate_breakdown(in_scope, old_refused, "false_rejects", "correct_answers"),
            "new_in_scope": _gate_breakdown(in_scope, new_refused, "false_rejects", "correct_answers"),
            "old_false_accept_rate": rate(out_scope, lambda r: not old_refused(r)),
            "new_false_accept_rate": rate(out_scope, lambda r: not new_refused(r)),
            "old_false_reject_rate": rate(in_scope, old_refused),
            "new_false_reject_rate": rate(in_scope, new_refused),
            "questions_missing_from_old_baseline": sorted(r["id"] for r in results if r["id"] not in old_baseline),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run eval/eval_set_v1.json against the live pipeline.")
    parser.add_argument("--namespace", required=True, help="Pinecone namespace to evaluate against, e.g. v2_5k.")
    parser.add_argument("--out", required=True, help="Label for output files, e.g. v2_5k.")
    parser.add_argument("--resume", action="store_true", help="Skip questions already present in the checkpoint file.")
    parser.add_argument("--dry-run", action="store_true", help="List questions and exit; no Claude/Pinecone calls.")
    parser.add_argument("--retrieval-only", action="store_true", help="Pinecone + local embeddings only, zero Claude calls. Reports provisional state, not generated answers.")
    parser.add_argument("--rerank", action="store_true", help="With --retrieval-only: also score the candidate pool with a local cross-encoder (no API cost).")
    parser.add_argument("--questions-path", default=QUESTIONS_PATH, help="Path to the question-set JSON, e.g. eval/eval_set_v2.json.")
    parser.add_argument(
        "--old-baseline",
        default="data/eval_v1_retrieval_only_rerank_v2_5k.json",
        help=(
            "Cached retrieval-only results (provisional_state per question id) used as the "
            "pre-judge similarity-floor baseline for summary comparison. Pass '' to skip the "
            "comparison. Ignored in --retrieval-only/--dry-run modes."
        ),
    )
    args = parser.parse_args()

    load_dotenv()

    questions = load_questions(args.questions_path)
    print(f"Loaded {len(questions)} questions "
          f"({sum(1 for q in questions if q['in_scope'])} in-scope, "
          f"{sum(1 for q in questions if not q['in_scope'])} out-of-scope). "
          f"Namespace: {args.namespace!r}")

    if args.dry_run:
        for q in questions:
            print(f"  [{q['topic']:30s}] id={q['id']} in_scope={q['in_scope']} {q['question'][:60]!r}")
        return

    if args.retrieval_only:
        results = []
        for q in questions:
            print(f"  [{q['topic']:30s}] id={q['id']} {q['question'][:70]!r}")
            results.append(run_one_question_retrieval_only(q, namespace=args.namespace, rerank=args.rerank))

        summary = summarize_retrieval_only(results)
        suffix = "_rerank" if args.rerank else ""
        summary_path = f"data/eval_v1_retrieval_only{suffix}_{args.out}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

        print(f"\nSaved {len(results)} question results + summary to {summary_path}.")
        print("\n=== SUMMARY (retrieval-only, no Claude calls) ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return

    checkpoint_path = f"data/eval_v1_checkpoint_{args.out}.jsonl"
    results_by_id = load_checkpoint(checkpoint_path) if args.resume else {}
    if results_by_id:
        print(f"Resuming: {len(results_by_id)} questions already completed in {checkpoint_path}.")

    checkpoint_f = open(checkpoint_path, "a", encoding="utf-8")
    try:
        for q in questions:
            if q["id"] in results_by_id:
                continue
            print(f"  [{q['topic']:30s}] id={q['id']} {q['question'][:70]!r}")
            record = run_one_question(q, namespace=args.namespace)
            results_by_id[q["id"]] = record
            checkpoint_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            checkpoint_f.flush()
    finally:
        checkpoint_f.close()

    results = sorted(results_by_id.values(), key=lambda r: r["id"])
    old_baseline = load_old_baseline(args.old_baseline)
    if args.old_baseline and not old_baseline:
        print(f"(No old-threshold baseline loaded from {args.old_baseline!r} — comparison section omitted.)")
    summary = summarize(results, old_baseline=old_baseline)

    summary_path = f"data/eval_v1_summary_{args.out}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} question results + summary to {summary_path}.")
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
