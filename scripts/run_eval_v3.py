"""
v3 eval on the 43-question dev set (eval/eval_set_v2.json), patient mode. Never touches the
held-out set.

    Run A: v2_5k + background, no scope judge (dense top-6 chunks on the original question)
    Run B: full v3 (query step, hybrid retrieval on v3_5k_recursive, full parents, ordering)

    python -m scripts.run_eval_v3 --config A
    python -m scripts.run_eval_v3 --config B
    python -m scripts.run_eval_v3 --smoke        # 3 unscored smoke questions, Run B

Resumable: each finished question is appended to <out>.partial.jsonl and skipped on restart, so a
completed question is never re-run. Query-step results are cached in data/query_step_cache.json.
Refuses to overwrite a finished output file.
"""
import argparse
import json
import os
import time

from dotenv import load_dotenv

from src.ingest.config import NAMESPACE_V2_5K, NAMESPACE_V3
from src.rag.pipeline import stream_answer_v3
from src.rag.query_rewriter import enable_disk_cache
from scripts.run_eval_v2_twostage import load_checkpoint, load_questions

QUESTIONS_PATH = "eval/eval_set_v2.json"
CONFIGS = {
    "A": {"namespace": NAMESPACE_V2_5K, "strategy": "dense_chunks", "out": "data/eval_v3_runA_v2_5k_noscope.json",
          "label": "Run A: v2_5k + background, no scope judge, dense top-6 chunks, patient mode"},
    "B": {"namespace": NAMESPACE_V3, "strategy": "hybrid", "out": "data/eval_v3_runB_full.json",
          "label": "Run B: full v3 (query step + hybrid on v3_5k_recursive + parents + ordering), patient mode"},
}
SMOKE_QUESTIONS = [
    {"id": "smoke1", "question": "What is the difference between type 1 and type 2 diabetes?"},
    {"id": "smoke2", "question": "What is HbA1c?"},
    {"id": "smoke3", "question": "What is insulin resistance?"},
]
SMOKE_OUT = "data/eval_v3_smoke.json"


def run_one(q, namespace, strategy, audience="patient"):
    t0 = time.monotonic()
    rec = {"id": q["id"], "topic": q.get("topic"), "in_scope": q.get("in_scope"), "question": q["question"],
           "reference_answer": q.get("reference_answer"), "status": None, "sources": [], "considered_sources": [],
           "sentence_records": [], "num_contradictions": 0, "error": None, "query_step": None, "retrieval": None,
           "evidence_verdict": None, "evidence_reason": None, "propositions": [], "population_mismatch": False,
           "refusal_message": None, "timing_ms": {}, "usage": None, "truncated": False, "model_generation": None,
           "audience": audience, "strategy": strategy, "namespace": namespace}
    for ev in stream_answer_v3(q["question"], namespace=namespace, audience=audience, retrieval_strategy=strategy):
        event, data = ev["event"], ev["data"]
        if event == "query":
            rec["query_step"] = data.get("query_step")
        elif event in ("sources", "refusal"):
            rec["status"] = data.get("status")
            for k in ("evidence_verdict", "evidence_reason", "propositions", "population_mismatch", "retrieval"):
                rec[k] = data.get(k, rec[k])
            rec["model_evidence"] = (data.get("judge") or {}).get("model")
            rec["timing_ms"].update(data.get("timing_ms") or {})
            if event == "sources":
                rec["sources"] = data.get("sources") or []
            else:
                rec["refusal_message"] = data.get("message")
                rec["considered_sources"] = data.get("considered_sources") or []
        elif event == "contradictions":
            rec["num_contradictions"] = len(data.get("contradictions") or [])
        elif event == "sentence":
            rec["sentence_records"].append(data)
        elif event == "done":
            rec["status"] = data.get("status", rec["status"])
            rec["timing_ms"].update(data.get("timing_ms") or {})
            rec["usage"], rec["truncated"], rec["model_generation"] = data.get("usage"), data.get("truncated"), data.get("model")
            rec["medication_note"] = data.get("medication_note")
        elif event == "error":
            rec["error"] = data.get("message")
    rec["generated_answer"] = " ".join(s["sentence"] for s in rec["sentence_records"]) or None
    # A judge or query step that failed on an API error falls back silently; count it as an error so a
    # resumed run retries it instead of recording it as a real result.
    api_failures = [r for r in (rec["evidence_reason"] or "", (rec["query_step"] or {}).get("reason") or "") if "API call failed" in r]
    if api_failures and not rec["error"]:
        rec["error"] = f"API failure: {api_failures[0]}"
    rec["wall_ms"] = round((time.monotonic() - t0) * 1000, 1)
    return rec


def run(questions, namespace, strategy, out, label):
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists; completed runs are never re-run.")
    checkpoint = out + ".partial.jsonl"
    done = load_checkpoint(checkpoint)
    results = []
    for q in questions:
        if q["id"] in done:
            results.append(done[q["id"]])
            continue
        print(f"  id={q['id']} {q['question'][:70]!r}", flush=True)
        rec = run_one(q, namespace, strategy)
        results.append(rec)
        if rec["error"]:
            print(f"    -> ERROR (not checkpointed): {rec['error']}", flush=True)
            continue
        with open(checkpoint, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"    -> {rec['status']}", flush=True)
    failed = [r["id"] for r in results if r["error"]]
    if failed:
        raise SystemExit(f"{len(failed)} failed: {failed}. Re-run the same command to retry only those.")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"pipeline": label, "namespace": namespace, "strategy": strategy, "audience": "patient", "results": results},
                  f, ensure_ascii=False, indent=2)
    os.remove(checkpoint)
    print(f"Saved {len(results)} results to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=sorted(CONFIGS))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    # One cache file per run, so runs going in parallel never overwrite each other's cache.
    cache = "data/query_step_cache.json" if args.config == "A" else "data/query_step_cache_B.json"
    print(f"Query-step cache: {enable_disk_cache(cache)} entries ({cache})")
    if args.smoke:
        b = CONFIGS["B"]
        run(SMOKE_QUESTIONS, b["namespace"], b["strategy"], SMOKE_OUT, "Smoke test (unscored), full v3, patient mode")
        return
    if not args.config:
        parser.error("--config or --smoke is required")
    c = CONFIGS[args.config]
    run(load_questions(QUESTIONS_PATH), c["namespace"], c["strategy"], c["out"], c["label"])


if __name__ == "__main__":
    main()
