"""
Domain evaluation harness: runs eval/test_questions.json (30 in-scope) and
eval/out_of_scope_questions.json (20 out-of-scope) through the real,
live pipeline (src.rag.chain.stream_answer) and scores groundedness, state
distribution, contradiction detection, latency, and real token usage.

Does NOT compute recall@5 — eval/test_questions.json has no ground-truth
expected_pmids per question yet, so it's reported as blocked rather than
faked.

Prints an estimated token cost for the whole batch (Pinecone-only, zero
Claude calls) before spending anything real. Checkpoints per-question
results to disk as it goes and can resume after an interruption.

Usage:
    python -m scripts.run_evaluation --dry-run          # cost estimate only, no Claude calls
    python -m scripts.run_evaluation --out v1_300        # run for real, label results v1_300
    python -m scripts.run_evaluation --out v1_300 --resume   # continue after an interruption
"""
import argparse
import json
import os
import statistics
import time

from dotenv import load_dotenv

from src.rag.chain import (
    determine_final_state,
    format_context,
    stream_answer,
    SYSTEM_PROMPT,
    CONTRA_START,
    CONTRA_END,
)
from src.rag.cost import estimate_tokens
from src.rag.retriever import retrieve_with_floor
from src.ingest.population import classify_population

IN_SCOPE_PATH = "eval/test_questions.json"
OUT_OF_SCOPE_PATH = "eval/out_of_scope_questions.json"


def load_questions():
    with open(IN_SCOPE_PATH, encoding="utf-8") as f:
        in_scope = [dict(q, expected_scope="in_scope") for q in json.load(f)["questions"]]
    with open(OUT_OF_SCOPE_PATH, encoding="utf-8") as f:
        out_of_scope = [dict(q, expected_scope="out_of_scope") for q in json.load(f)["questions"]]
    return in_scope + out_of_scope


def estimate_batch_cost(questions):
    """
    Pinecone-only pre-pass: for each question, run real retrieval (free)
    and estimate the prompt size it would produce, without calling Claude.
    Questions that would refuse pre-generation (out_of_scope) cost nothing.
    """
    total_tokens = 0
    would_call = 0
    for q in questions:
        decision = retrieve_with_floor(q["question"])
        if decision.state == "out_of_scope":
            continue
        chunks = decision.surviving_chunks
        requested_population = classify_population(q["question"])
        context_str = format_context(chunks)
        system_prompt = SYSTEM_PROMPT.format(
            requested_population=requested_population,
            retrieved_populations="",
            contra_start=CONTRA_START,
            contra_end=CONTRA_END,
            context=context_str,
        )
        total_tokens += estimate_tokens(system_prompt + " " + q["question"])
        would_call += 1
    return total_tokens, would_call


def run_one_question(q):
    """
    Consume stream_answer() fully for one question and reduce it to a flat
    result record. Runs the real pipeline — a real Claude call unless the
    question refuses pre-generation.
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

    for ev in stream_answer(q["question"]):
        event, data = ev["event"], ev["data"]
        if event == "sources":
            sources = data.get("sources", [])
        elif event == "contradictions":
            contradictions = data.get("contradictions", [])
        elif event == "sentence":
            sentence_records.append(data)
        elif event == "done":
            final_state = data.get("state")
            timing_ms = data.get("timing_ms", {})
            usage = data.get("usage")
            truncated = data.get("truncated", False)
        elif event == "refusal":
            final_state = data.get("state", "out_of_scope")
            timing_ms = data.get("timing_ms", {})
            refusal_message = data.get("message")
        elif event == "error":
            error = data.get("message")

    wall_ms = round((time.monotonic() - t0) * 1000, 1)

    # source_type is now computed server-side per sentence (Phase 2) —
    # use it directly instead of re-deriving it from the sources list, so
    # the eval harness's numbers can never drift from the product's own
    # authoritative definition of "what tier backs this sentence."
    evidence_supported = sum(1 for s in sentence_records if s.get("source_type") == "evidence")
    background_supported = sum(1 for s in sentence_records if s.get("source_type") == "background")
    unsupported = sum(1 for s in sentence_records if not s.get("chunk_ids"))

    total_sentences = len(sentence_records)

    return {
        "id": q["id"],
        "question": q["question"],
        "category": q.get("category"),
        "expected_scope": q["expected_scope"],
        "final_state": final_state,
        "error": error,
        "refusal_message": refusal_message,
        "num_sources": len(sources),
        "num_contradictions": len(contradictions),
        "total_sentences": total_sentences,
        "evidence_supported_sentences": evidence_supported,
        "background_supported_sentences": background_supported,
        "unsupported_sentences": unsupported,
        "timing_ms": timing_ms,
        "wall_ms": wall_ms,
        "usage": usage,
        "truncated": truncated,
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
                done[(rec["expected_scope"], rec["id"])] = rec
    return done


def summarize(results):
    in_scope = [r for r in results if r["expected_scope"] == "in_scope"]
    out_scope = [r for r in results if r["expected_scope"] == "out_of_scope"]

    def rate(items, pred):
        return round(sum(1 for i in items if pred(i)) / len(items), 4) if items else None

    def mean(values):
        values = [v for v in values if v is not None]
        return round(statistics.mean(values), 2) if values else None

    generated = [r for r in results if r["final_state"] not in ("out_of_scope", None) and r["error"] is None]
    total_sentences_all = sum(r["total_sentences"] for r in generated)
    evidence_sentences_all = sum(r["evidence_supported_sentences"] for r in generated)
    background_sentences_all = sum(r["background_supported_sentences"] for r in generated)
    unsupported_all = sum(r["unsupported_sentences"] for r in generated)

    return {
        "n_in_scope": len(in_scope),
        "n_out_of_scope": len(out_scope),
        "out_of_scope_refusal_rate": rate(out_scope, lambda r: r["final_state"] == "out_of_scope"),
        "in_scope_answer_rate": rate(in_scope, lambda r: r["final_state"] in ("answered", "answered_low_confidence")),
        "no_evidence_for_claim_rate_in_scope": rate(in_scope, lambda r: r["final_state"] == "no_evidence_for_claim"),
        "answered_low_confidence_rate_in_scope": rate(in_scope, lambda r: r["final_state"] == "answered_low_confidence"),
        "groundedness_overall": round((evidence_sentences_all + background_sentences_all) / total_sentences_all, 4) if total_sentences_all else None,
        "groundedness_evidence_only": round(evidence_sentences_all / total_sentences_all, 4) if total_sentences_all else None,
        "groundedness_background_only": round(background_sentences_all / total_sentences_all, 4) if total_sentences_all else None,
        "total_sentences": total_sentences_all,
        "unsupported_sentences": unsupported_all,
        "total_contradictions_detected": sum(r["num_contradictions"] for r in results),
        "truncated_responses": sum(1 for r in results if r.get("truncated")),
        "mean_latency_ms_out_of_scope": mean([r["timing_ms"].get("retrieval") for r in out_scope]),
        "mean_latency_ms_generated_total": mean([r["timing_ms"].get("total") for r in generated]),
        "mean_input_tokens": mean([r["usage"]["input_tokens"] for r in generated if r["usage"]]),
        "mean_output_tokens": mean([r["usage"]["output_tokens"] for r in generated if r["usage"]]),
        "errors": sum(1 for r in results if r["error"]),
        "recall_at_5": "BLOCKED: eval/test_questions.json has no ground-truth expected_pmids per question",
    }


def main():
    parser = argparse.ArgumentParser(description="Run the domain evaluation against the live pipeline.")
    parser.add_argument("--out", default="baseline", help="Label for output files, e.g. v1_300 or v2_5k.")
    parser.add_argument("--dry-run", action="store_true", help="Print the cost estimate only; no Claude calls.")
    parser.add_argument("--resume", action="store_true", help="Skip questions already present in the checkpoint file.")
    args = parser.parse_args()

    load_dotenv()

    questions = load_questions()
    print(f"Loaded {len(questions)} questions ({sum(1 for q in questions if q['expected_scope']=='in_scope')} in-scope, "
          f"{sum(1 for q in questions if q['expected_scope']=='out_of_scope')} out-of-scope).")

    print("\nEstimating batch cost (Pinecone-only, zero Claude calls)...")
    est_tokens, would_call = estimate_batch_cost(questions)
    print(f"[cost] ~{est_tokens} estimated input tokens across {would_call} questions expected to reach generation "
          f"({len(questions) - would_call} expected to refuse pre-generation, free).")

    if args.dry_run:
        print("Dry run requested — stopping before any Claude call.")
        return

    checkpoint_path = f"data/eval_checkpoint_{args.out}.jsonl"
    results_by_id = load_checkpoint(checkpoint_path) if args.resume else {}
    if results_by_id:
        print(f"Resuming: {len(results_by_id)} questions already completed in {checkpoint_path}.")

    checkpoint_f = open(checkpoint_path, "a", encoding="utf-8")
    try:
        for q in questions:
            key = (q["expected_scope"], q["id"])
            if key in results_by_id:
                continue
            print(f"  [{q['expected_scope']:13s}] id={q['id']} {q['question'][:70]!r}")
            record = run_one_question(q)
            results_by_id[key] = record
            checkpoint_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            checkpoint_f.flush()
    finally:
        checkpoint_f.close()

    results = list(results_by_id.values())
    summary = summarize(results)

    summary_path = f"data/eval_summary_{args.out}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} question results + summary to {summary_path}.")
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
