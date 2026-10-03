"""
Latency measurement: runs dev-set questions through the production pipeline (patient mode, default
namespace and settings, no eval caches) and records time per stage.

    python -m scripts.measure_latency --label before --out data/latency_before.json
    JUDGE_PASSAGES=4 python -m scripts.measure_latency --label after_judge4 --out data/latency_after_judge4.json

Default questions: dev ids 2, 8, 12, 15, 22 (eval/eval_set_v2.json): two fully answered, two that
triggered the patient-mode drug check in Run B (8, 12), one partial. Never the held-out set.
"""
import argparse
import json
import statistics
import time

from dotenv import load_dotenv

from scripts.run_eval_v2_twostage import load_questions
from src.rag.pipeline import stream_answer_v3

DEFAULT_IDS = [2, 8, 12, 15, 22]
STAGES = [("query", "rewrite"), ("retrieval", "search"), ("evidence", "judge"), ("generation", "generation"),
          ("drug_check", "drug check"), ("citation_checks", "citation checks"), ("verifier", "verifier")]


def measure(question):
    t0 = time.monotonic()
    rec = {"status": None, "timing_ms": {}, "drug_check": None, "usage": None}
    for ev in stream_answer_v3(question):
        d = ev["data"]
        if ev["event"] in ("sources", "refusal"):
            rec["status"] = d.get("status")
            rec["timing_ms"].update(d.get("timing_ms") or {})
        elif ev["event"] == "done":
            rec["timing_ms"].update(d.get("timing_ms") or {})
            rec["drug_check"], rec["usage"] = d.get("drug_check"), d.get("usage")
        elif ev["event"] == "error":
            rec["error"] = d.get("message")
    rec["wall_ms"] = round((time.monotonic() - t0) * 1000, 1)
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default=",".join(map(str, DEFAULT_IDS)))
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    load_dotenv()
    wanted = [int(i) for i in args.ids.split(",")]
    questions = {q["id"]: q for q in load_questions("eval/eval_set_v2.json")}
    rows = []
    for i in wanted:
        rec = {"id": i, **measure(questions[i]["question"])}
        rows.append(rec)
        t = rec["timing_ms"]
        print(f"id {i}: {rec['status']} wall {rec['wall_ms']/1000:.1f}s | " +
              " ".join(f"{name} {t[k]/1000:.1f}s" for k, name in STAGES if t.get(k) is not None), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"label": args.label, "results": rows}, f, indent=1)
    print("\nmean per stage (s):", {name: round(statistics.mean([r["timing_ms"].get(k, 0) or 0 for r in rows]) / 1000, 2)
                                    for k, name in STAGES})
    print("mean wall (s):", round(statistics.mean(r["wall_ms"] for r in rows) / 1000, 2))
    print("drug check regenerated:", sum(1 for r in rows if (r["drug_check"] or {}).get("regenerated")), "of", len(rows))


if __name__ == "__main__":
    main()
