"""
Minimal end-to-end smoke test against the live pipeline (real Pinecone +
real Claude calls) — proves the generation path works, not a measurement
run. Exactly 3 fixed questions, pulled by id from the frozen
eval/eval_set_v2.json so they're traceable back to that set, not
hand-typed here:

  1. id=1  (in-scope, diet_nutrition)              — expect: answered, cited, namespace v2_5k
  2. id=43 (out_of_scope_unrelated, "car loan")     — expect: refusal at the retrieval floor, zero Claude cost
  3. id=25 (out_of_scope_diabetes_adjacent, retinopathy) — expect: clears the floor, reaches generation

Every raw SSE event list is cached verbatim to data/smoke_test_<id>_<slug>.json
so this never has to be re-run to inspect the output. Reports real token
usage (response.usage from Claude, not an estimate) and cost at Sonnet 5
pricing ($2.00 / $10.00 per 1M input/output tokens).

Usage:
    python -m scripts.smoke_test_v2_5k
"""
import json
import os
import re

from dotenv import load_dotenv

from src.rag.chain import stream_answer

EVAL_SET_PATH = "eval/eval_set_v2.json"
NAMESPACE = "v2_5k"
QUESTION_IDS = [1, 43, 25]

# Sonnet 5 pricing, per 1M tokens (see model pricing table).
INPUT_PRICE_PER_MTOK = 2.00
OUTPUT_PRICE_PER_MTOK = 10.00


def slugify(text, max_len=40):
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_len]


def load_question(qid):
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    match = next((q for q in questions if q["id"] == qid), None)
    if match is None:
        raise SystemExit(f"No question with id={qid} in {EVAL_SET_PATH}")
    return match


def run_one(q):
    events = []
    for ev in stream_answer(q["question"], namespace=NAMESPACE):
        events.append(ev)

    slug = slugify(q["question"])
    out_path = f"data/smoke_test_{q['id']}_{slug}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"question": q, "namespace": NAMESPACE, "events": events}, f, ensure_ascii=False, indent=2)

    done = next((e["data"] for e in events if e["event"] == "done"), None)
    refusal = next((e["data"] for e in events if e["event"] == "refusal"), None)
    error = next((e["data"] for e in events if e["event"] == "error"), None)
    sources = next((e["data"] for e in events if e["event"] == "sources"), None)

    usage = (done or {}).get("usage")
    input_tokens = usage["input_tokens"] if usage else 0
    output_tokens = usage["output_tokens"] if usage else 0
    cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_MTOK + (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MTOK

    reported_namespace = None
    if sources:
        reported_namespace = sources.get("score_distribution", {}).get("namespace")
    elif refusal:
        reported_namespace = refusal.get("score_distribution", {}).get("namespace")

    return {
        "id": q["id"],
        "question": q["question"],
        "cached_to": out_path,
        "final_state": (done or refusal or {}).get("state") or (refusal or {}).get("state"),
        "reached_generation": done is not None,
        "error": (error or {}).get("message"),
        "namespace_in_payload": reported_namespace,
        "num_sources": len(sources["sources"]) if sources else 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(cost, 6),
    }


def main():
    load_dotenv()
    print(f"Running {len(QUESTION_IDS)} smoke-test questions against namespace {NAMESPACE!r}. No eval, no batch.\n")

    results = []
    for qid in QUESTION_IDS:
        q = load_question(qid)
        print(f"[{qid}] {q['question']}")
        r = run_one(q)
        results.append(r)
        print(f"    state={r['final_state']!r} reached_generation={r['reached_generation']} "
              f"namespace_in_payload={r['namespace_in_payload']!r} sources={r['num_sources']} "
              f"tokens(in/out)={r['input_tokens']}/{r['output_tokens']} "
              f"cost=${r['estimated_cost_usd']:.6f} -> {r['cached_to']}\n")

    total_cost = sum(r["estimated_cost_usd"] for r in results)
    total_in = sum(r["input_tokens"] for r in results)
    total_out = sum(r["output_tokens"] for r in results)
    print("=== SUMMARY ===")
    for r in results:
        print(f"  id={r['id']:<3} state={r['final_state']:<24} in={r['input_tokens']:<6} out={r['output_tokens']:<6} cost=${r['estimated_cost_usd']:.6f}")
    print(f"  TOTAL: in={total_in} out={total_out} cost=${total_cost:.6f}")


if __name__ == "__main__":
    main()
