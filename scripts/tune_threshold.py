"""
Empirically tune SIMILARITY_FLOOR instead of guessing it.

Retrieves a wide candidate pool (RETRIEVAL_CANDIDATE_K) once per question —
for both the in-scope evaluation set (eval/test_questions.json) and a
deliberately unrelated out-of-scope set (eval/out_of_scope_questions.json,
covering unrelated conditions, general trivia, and nonsense) — then sweeps
a range of floor values purely against the cached scores. No Claude calls
are made: the floor decision is retrieval-score logic only, so this script
costs nothing but Pinecone queries.

For each floor value, reports:
  - in-scope answer rate: fraction of the 30 in-scope questions where at
    least one chunk clears the floor (state != "refused")
  - out-of-scope refusal rate: fraction of the 20 out-of-scope questions
    correctly refused (state == "refused")

Saves the full sweep to data/threshold_sweep.csv and prints a recommended
floor: the value that maximizes out-of-scope refusal rate while keeping
in-scope answer rate at its maximum observed level.

Usage:
    python -m scripts.tune_threshold
    python -m scripts.tune_threshold --start 0.30 --stop 0.70 --step 0.01
"""
import argparse
import csv
import json
import os

from dotenv import load_dotenv

from src.rag.retriever import decide_retrieval_state, get_candidate_k, retrieve

IN_SCOPE_PATH = "eval/test_questions.json"
OUT_OF_SCOPE_PATH = "eval/out_of_scope_questions.json"
SWEEP_CSV_PATH = "data/threshold_sweep.csv"


def load_questions(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["questions"]


def fetch_candidates(questions, k: int):
    """Retrieve once per question; every floor value below reuses these scores."""
    results = []
    for q in questions:
        chunks = retrieve(q["question"], k=k)
        results.append({"id": q["id"], "question": q["question"], "chunks": chunks})
        print(f"  fetched candidates for [{q.get('category', '?')}] {q['question'][:70]!r}")
    return results


def sweep(in_scope_candidates, out_of_scope_candidates, floors, margin: float):
    rows = []
    for floor in floors:
        in_scope_answered = sum(
            1 for r in in_scope_candidates
            if decide_retrieval_state(r["chunks"], floor, margin).state != "refused"
        )
        out_of_scope_refused = sum(
            1 for r in out_of_scope_candidates
            if decide_retrieval_state(r["chunks"], floor, margin).state == "refused"
        )
        n_in, n_out = len(in_scope_candidates), len(out_of_scope_candidates)
        rows.append({
            "floor": round(floor, 3),
            "in_scope_answer_rate": round(in_scope_answered / n_in, 4),
            "in_scope_answered_count": in_scope_answered,
            "in_scope_total": n_in,
            "out_of_scope_refusal_rate": round(out_of_scope_refused / n_out, 4),
            "out_of_scope_refused_count": out_of_scope_refused,
            "out_of_scope_total": n_out,
        })
    return rows


def recommend_floor(rows):
    """
    Pick the highest floor that still keeps in-scope answer rate at its
    best observed level, breaking ties toward higher out-of-scope refusal.
    This is "push the floor up for cleaner refusals until in-scope answers
    would start to drop" — not an arbitrary midpoint.
    """
    best_in_scope_rate = max(r["in_scope_answer_rate"] for r in rows)
    at_best = [r for r in rows if r["in_scope_answer_rate"] == best_in_scope_rate]
    best_refusal_at_best = max(r["out_of_scope_refusal_rate"] for r in at_best)
    candidates = [r for r in at_best if r["out_of_scope_refusal_rate"] == best_refusal_at_best]
    return max(candidates, key=lambda r: r["floor"])


def main():
    parser = argparse.ArgumentParser(description="Sweep SIMILARITY_FLOOR against real retrieval scores.")
    parser.add_argument("--start", type=float, default=0.30)
    parser.add_argument("--stop", type=float, default=0.70)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=0.05, help="LOW_CONFIDENCE_MARGIN (doesn't affect refusal, only answered vs low-confidence).")
    args = parser.parse_args()

    load_dotenv()

    candidate_k = get_candidate_k()
    print(f"Candidate pool size: {candidate_k}")

    print(f"\nFetching in-scope candidates from {IN_SCOPE_PATH} ...")
    in_scope = fetch_candidates(load_questions(IN_SCOPE_PATH), candidate_k)

    print(f"\nFetching out-of-scope candidates from {OUT_OF_SCOPE_PATH} ...")
    out_of_scope = fetch_candidates(load_questions(OUT_OF_SCOPE_PATH), candidate_k)

    floors = []
    f = args.start
    while f <= args.stop + 1e-9:
        floors.append(round(f, 4))
        f += args.step

    print(f"\nSweeping {len(floors)} floor values from {args.start} to {args.stop} (step {args.step}) ...")
    rows = sweep(in_scope, out_of_scope, floors, args.margin)

    os.makedirs(os.path.dirname(SWEEP_CSV_PATH) or ".", exist_ok=True)
    with open(SWEEP_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved sweep to {SWEEP_CSV_PATH}")

    print("\nfloor  | in-scope answer rate      | out-of-scope refusal rate")
    print("-------|---------------------------|---------------------------")
    for r in rows:
        print(
            f"{r['floor']:.3f}  | {r['in_scope_answer_rate']:.3f} ({r['in_scope_answered_count']}/{r['in_scope_total']})"
            f"{'':>6}| {r['out_of_scope_refusal_rate']:.3f} ({r['out_of_scope_refused_count']}/{r['out_of_scope_total']})"
        )

    rec = recommend_floor(rows)
    print(
        f"\nRecommended SIMILARITY_FLOOR = {rec['floor']}"
        f"\n  in-scope answer rate:      {rec['in_scope_answer_rate']:.3f} ({rec['in_scope_answered_count']}/{rec['in_scope_total']})"
        f"\n  out-of-scope refusal rate: {rec['out_of_scope_refusal_rate']:.3f} ({rec['out_of_scope_refused_count']}/{rec['out_of_scope_total']})"
        f"\n  (highest floor that keeps in-scope answers at their best observed rate)"
    )


if __name__ == "__main__":
    main()
