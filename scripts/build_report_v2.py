"""
Computes every figure in eval/REPORT_twostage_judge.md and
eval/relabel_proposals.md from files in data/ and eval/ — never hand-typed
— with internal-consistency assertions, matching the same discipline as
the previous eval report's build script (see CHANGES.md for why that
discipline exists: an earlier draft of that report had a hand-counting
error a computed assertion would have caught immediately).

Reads (never writes to eval/eval_set_v2.json — the frozen question file):
    eval/eval_set_v2.json                              — ground-truth labels
    data/eval_v2_judge_twostage_v2_5k.json              — new two-stage run (Step 9)
    data/eval_v2_judge_twostage_v2_5k_faithfulness.json — grader verdicts for it
    data/eval_v1_summary_v2_5k_judge_v2.json            — previous (boolean-judge) run, for comparison
    data/eval_v1_old_pipeline_full_v2_5k.json           — original (floor-only) run, for comparison

Writes:
    data/eval_v2_report_figures.json  — every figure, as JSON
    eval/relabel_proposals.md         — adjacent questions with a SUPPORTED answer
    eval/REPORT_twostage_judge.md     — the full report

Usage:
    python -m scripts.build_report_v2
"""
import json
import math
import statistics

EVAL_SET_PATH = "eval/eval_set_v2.json"
NEW_PATH = "data/eval_v2_judge_twostage_v2_5k.json"
NEW_FAITH_PATH = "data/eval_v2_judge_twostage_v2_5k_faithfulness.json"
PREV_JUDGE_PATH = "data/eval_v1_summary_v2_5k_judge_v2.json"
OLD_FLOOR_PATH = "data/eval_v1_old_pipeline_full_v2_5k.json"

FIGURES_OUT = "data/eval_v2_report_figures.json"
RELABEL_OUT = "eval/relabel_proposals.md"
REPORT_OUT = "eval/REPORT_twostage_judge.md"

ANSWERED_STATUSES = {"answered", "answered_partial", "answered_adjacent"}
CALLOUT_IDS = [1, 7, 10, 18, 20, 22, 25, 33, 37]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_eval_set():
    return {q["id"]: q for q in load_json(EVAL_SET_PATH)["questions"]}


def load_results(path):
    return {r["id"]: r for r in load_json(path)["results"]}


def eval_label(q):
    """in_scope / adjacent / unrelated, from the frozen eval set's own labels."""
    if q["in_scope"]:
        return "in_scope"
    if q["topic"] == "out_of_scope_diabetes_adjacent":
        return "adjacent"
    return "unrelated"


def wilson_ci(successes, n, z=1.96):
    """
    Wilson score interval for a binomial proportion — noticeably better
    calibrated than a normal approximation at the small n (2, 17, 24) this
    eval set has. Returns (point_estimate, lo, hi), all rounded to 4dp.
    Returns (None, None, None) for n=0.
    """
    if n == 0:
        return None, None, None
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return round(p, 4), round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)


def is_answered_status(status):
    return status in ANSWERED_STATUSES


def old_floor_is_answered(record):
    return record["final_state"] in ("answered", "answered_low_confidence") and not record.get("error")


def prev_judge_is_answered(record):
    return record["final_state"] in ("answered", "answered_low_confidence") and not record.get("error")


def build_core(eval_set, new_p):
    ids_all = sorted(eval_set)
    by_label = {"in_scope": [], "adjacent": [], "unrelated": []}
    for i in ids_all:
        by_label[eval_label(eval_set[i])].append(i)
    return {"ids_all": ids_all, "by_label": by_label}


def run_assertions(eval_set, core, new_p, faith):
    log = []

    def check(label, cond):
        status = "PASS" if cond else "FAIL"
        log.append(f"[{status}] {label}")
        if not cond:
            raise AssertionError(f"Consistency assertion failed: {label}")

    check("43 questions total", len(eval_set) == 43)
    check("24 in_scope", len(core["by_label"]["in_scope"]) == 24)
    check("17 adjacent", len(core["by_label"]["adjacent"]) == 17)
    check("2 unrelated", len(core["by_label"]["unrelated"]) == 2)
    check(
        "labels partition all 43 with no overlap",
        sorted(core["by_label"]["in_scope"] + core["by_label"]["adjacent"] + core["by_label"]["unrelated"]) == core["ids_all"],
    )
    check("new run covers all 43 ids", sorted(new_p.keys()) == core["ids_all"])

    # Every status must be one of the five defined statuses, or None with an error.
    valid_statuses = {"answered", "answered_partial", "in_scope_no_evidence", "answered_adjacent", "out_of_scope"}
    for i in core["ids_all"]:
        s = new_p[i]["status"]
        check(f"id {i} has a valid status or a recorded error", s in valid_statuses or (s is None and new_p[i].get("error")))

    # in_scope_no_evidence must never say "out of scope" in its own message.
    for i in core["ids_all"]:
        r = new_p[i]
        if r["status"] == "in_scope_no_evidence":
            check(f"id {i} in_scope_no_evidence message never says 'out of scope'", "out of scope" not in (r.get("refusal_message") or "").lower())

    # Every graded id in the faithfulness file must actually be an answered case in new_p.
    graded_ids = {g["id"] for g in faith["results"]}
    answered_ids = {i for i in core["ids_all"] if is_answered_status(new_p[i]["status"])}
    check("every graded id was actually answered", graded_ids <= answered_ids)
    check("every answered id was graded (grader ran on every answered case, not a subset)", answered_ids <= graded_ids)

    return log


def per_label_stats(core, new_p, old_p, prev_p):
    stats = {}
    for label, ids in core["by_label"].items():
        n = len(ids)
        new_answered = [i for i in ids if is_answered_status(new_p[i]["status"])]
        new_no_evidence = [i for i in ids if new_p[i]["status"] == "in_scope_no_evidence"]
        new_refused = [i for i in ids if new_p[i]["status"] == "out_of_scope"]
        old_answered = [i for i in ids if old_floor_is_answered(old_p[i])]
        prev_answered = [i for i in ids if prev_judge_is_answered(prev_p[i])]

        stats[label] = {
            "n": n,
            "new_answer_rate": wilson_ci(len(new_answered), n),
            "new_in_scope_no_evidence_rate": wilson_ci(len(new_no_evidence), n),
            "new_refusal_rate": wilson_ci(len(new_refused), n),
            "old_floor_answer_rate": wilson_ci(len(old_answered), n),
            "prev_judge_answer_rate": wilson_ci(len(prev_answered), n),
            "new_answered_ids": sorted(new_answered),
            "new_no_evidence_ids": sorted(new_no_evidence),
            "new_refused_ids": sorted(new_refused),
        }
    return stats


def confusion_matrix(eval_set, core, new_p):
    scopes = ["in_charter", "adjacent", "unrelated"]
    labels = ["in_scope", "adjacent", "unrelated"]
    matrix = {s: {l: [] for l in labels} for s in scopes}
    for i in core["ids_all"]:
        scope = new_p[i]["scope"]
        label = eval_label(eval_set[i])
        if scope in scopes:
            matrix[scope][label].append(i)
    return matrix


def faithfulness_distribution(ids, faith_by_id):
    counts = {"SUPPORTED": 0, "PARTIALLY_SUPPORTED": 0, "UNSUPPORTED": 0, "not_graded": 0}
    for i in ids:
        v = faith_by_id.get(i)
        overall = v["verdict"].get("overall") if v else None
        counts[overall if overall in counts else "not_graded"] += 1
    return counts


def latency_and_cost(new_p, old_p):
    def mean_p95(values):
        values = [v for v in values if v is not None]
        if not values:
            return None, None
        return round(statistics.mean(values), 1), round(sorted(values)[max(0, int(len(values) * 0.95) - 1)], 1)

    new_scope_ms = [r["timing_ms"].get("scope") for r in new_p.values()]
    new_evidence_ms = [r["timing_ms"].get("evidence") for r in new_p.values()]
    new_retrieval_ms = [r["timing_ms"].get("retrieval") for r in new_p.values()]
    new_gen_ms = [r["timing_ms"].get("generation") for r in new_p.values() if is_answered_status(r["status"])]
    new_total_ms = [r["timing_ms"].get("total") for r in new_p.values()]
    old_gen_ms = [r["timing_ms"].get("generation") for r in old_p.values() if old_floor_is_answered(r)]

    new_input_tok = [r["usage"]["input_tokens"] for r in new_p.values() if r.get("usage")]
    new_output_tok = [r["usage"]["output_tokens"] for r in new_p.values() if r.get("usage")]
    old_input_tok = [r["usage"]["input_tokens"] for r in old_p.values() if r.get("usage")]
    old_output_tok = [r["usage"]["output_tokens"] for r in old_p.values() if r.get("usage")]

    return {
        "new_scope_ms": mean_p95(new_scope_ms),
        "new_retrieval_ms": mean_p95(new_retrieval_ms),
        "new_evidence_ms": mean_p95(new_evidence_ms),
        "new_generation_ms": mean_p95(new_gen_ms),
        "new_total_ms": mean_p95(new_total_ms),
        "old_generation_ms": mean_p95(old_gen_ms),
        "new_mean_input_tokens": round(statistics.mean(new_input_tok), 1) if new_input_tok else None,
        "new_mean_output_tokens": round(statistics.mean(new_output_tok), 1) if new_output_tok else None,
        "old_mean_input_tokens": round(statistics.mean(old_input_tok), 1) if old_input_tok else None,
        "old_mean_output_tokens": round(statistics.mean(old_output_tok), 1) if old_output_tok else None,
    }


def model_versions(new_p):
    return {
        "scope_models": sorted({r["model_scope"] for r in new_p.values() if r.get("model_scope")}),
        "evidence_models": sorted({r["model_evidence"] for r in new_p.values() if r.get("model_evidence")}),
        "generation_models": sorted({r["model_generation"] for r in new_p.values() if r.get("model_generation")}),
    }


def build_relabel_proposals(eval_set, core, new_p, faith_by_id):
    """Adjacent questions where the new pipeline produced an answer graded SUPPORTED — candidates the corpus can actually answer despite being outside the four charter areas."""
    candidates = []
    for i in core["by_label"]["adjacent"]:
        r = new_p[i]
        if not is_answered_status(r["status"]):
            continue
        g = faith_by_id.get(i)
        overall = g["verdict"].get("overall") if g else None
        if overall == "SUPPORTED":
            candidates.append(i)
    # Task explicitly wants 25, 33, 37 first if present, then the rest by id.
    priority = [i for i in (25, 33, 37) if i in candidates]
    rest = sorted(i for i in candidates if i not in priority)
    return priority + rest


def render_relabel_md(eval_set, new_p, candidate_ids):
    lines = [
        "# Relabel proposals",
        "",
        "Diabetes-adjacent questions (outside the four charter areas) where the corpus produced an answer the",
        "faithfulness grader rated SUPPORTED — meaning the corpus actually has good evidence for these, even though",
        "they fall outside `config/scope_charter.md`'s four core areas. **These are proposals for the maintainer,",
        "not changes** — `eval/eval_set_v2.json`'s labels are untouched.",
        "",
    ]
    if not candidate_ids:
        lines.append("No candidates found in this run.")
    for i in candidate_ids:
        r = new_p[i]
        q = eval_set[i]
        supporting = "\n".join(
            f"- [{j+1}] ({s.get('source_type')}) {s.get('title', '')} — {s.get('excerpt', '')[:220]}"
            for j, s in enumerate(r.get("sources") or [])
            if j + 1 in {c for sr in (r.get("sentence_records") or []) for c in (sr.get("chunk_ids") or [])}
        )
        lines += [
            f"## id {i} — {q['topic']} / {q.get('sub_area', '')}",
            "",
            f"**Question:** {q['question']}",
            "",
            f"**Answer:** {r.get('generated_answer', '')}",
            "",
            "**Supporting passages:**",
            supporting or "(none captured)",
            "",
            "**For relabeling to in_charter:** the corpus has direct, faithful evidence for this question — a user asking it gets a real answer today, and the charter's own four areas are somewhat arbitrary boundaries that don't reflect what the corpus can actually do.",
            "",
            "**Against relabeling:** the charter defines scope by subject matter, not by what one snapshot of the corpus happens to contain — a future re-ingestion could easily remove this coverage, and broadening the charter to match today's incidental corpus contents risks scope creep untethered from the product's actual four-area design intent.",
            "",
        ]
    return "\n".join(lines)


def render_table_row(eval_set, core, new_p, old_p, prev_p, faith_by_id, i):
    q = eval_set[i]
    label = eval_label(q)
    r = new_p[i]
    old_status = "answered" if old_floor_is_answered(old_p[i]) else old_p[i]["final_state"]
    prev_status = "answered" if prev_judge_is_answered(prev_p[i]) else prev_p[i]["final_state"]
    g = faith_by_id.get(i)
    faithfulness = g["verdict"].get("overall") if g else ("n/a" if not is_answered_status(r["status"]) else "not graded")
    return (
        f"| {i} | {label} | {q['topic']} | {r.get('top_score') or 0:.3f} | {old_status} | {prev_status} | "
        f"{r['scope']} | {r['evidence_verdict']} | {r['status']} | {faithfulness} |"
    )


def main():
    eval_set = load_eval_set()
    new_p = load_results(NEW_PATH)
    old_p = load_results(OLD_FLOOR_PATH)
    prev_p = load_results(PREV_JUDGE_PATH)
    faith = load_json(NEW_FAITH_PATH)
    faith_by_id = {g["id"]: g for g in faith["results"]}

    core = build_core(eval_set, new_p)
    assertion_log = run_assertions(eval_set, core, new_p, faith)
    for line in assertion_log:
        print(line)

    stats = per_label_stats(core, new_p, old_p, prev_p)
    confusion = confusion_matrix(eval_set, core, new_p)
    faith_dist = {label: faithfulness_distribution(ids, faith_by_id) for label, ids in core["by_label"].items()}
    lat_cost = latency_and_cost(new_p, old_p)
    models = model_versions(new_p)
    relabel_candidates = build_relabel_proposals(eval_set, core, new_p, faith_by_id)

    figures = {
        "per_label_stats": stats,
        "confusion_matrix": {s: {l: {"n": len(ids), "ids": sorted(ids)} for l, ids in labels.items()} for s, labels in confusion.items()},
        "faithfulness_distribution": faith_dist,
        "latency_and_cost": lat_cost,
        "model_versions": models,
        "relabel_candidate_ids": relabel_candidates,
    }
    with open(FIGURES_OUT, "w", encoding="utf-8") as f:
        json.dump(figures, f, ensure_ascii=False, indent=2)
    print(f"\nWrote figures to {FIGURES_OUT}")

    with open(RELABEL_OUT, "w", encoding="utf-8") as f:
        f.write(render_relabel_md(eval_set, new_p, relabel_candidates))
    print(f"Wrote {RELABEL_OUT}")

    # Per-question table + callouts + report assembled in build_full_report_markdown.py-equivalent below.
    table_rows = [render_table_row(eval_set, core, new_p, old_p, prev_p, faith_by_id, i) for i in core["ids_all"]]
    callout_rows = [render_table_row(eval_set, core, new_p, old_p, prev_p, faith_by_id, i) for i in CALLOUT_IDS if i in new_p]

    report = build_report_markdown(eval_set, core, stats, confusion, faith_dist, lat_cost, models, table_rows, callout_rows, assertion_log)
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote {REPORT_OUT}")


def _fmt_ci(triple):
    p, lo, hi = triple
    if p is None:
        return "n/a"
    return f"{p*100:.1f}% (95% CI {lo*100:.1f}–{hi*100:.1f}%)"


def build_report_markdown(eval_set, core, stats, confusion, faith_dist, lat_cost, models, table_rows, callout_rows, assertion_log):
    lines = []
    lines.append("# Two-stage answerability judge — eval report")
    lines.append("")
    lines.append("Single run, `eval/eval_set_v2.json` (43 questions, unmodified), production pipeline "
                 "(`src.rag.chain.stream_answer`, no diagnostic overrides). Every number below is computed by "
                 "`scripts/build_report_v2.py` from files in `data/` and `eval/` — see that script for the exact "
                 "derivation, and §7 for the consistency assertions it ran before writing anything.")
    lines.append("")

    lines.append("## 1. What changed and why")
    lines.append("")
    lines.append("The single boolean answerability judge (`src/rag/judge.py`) conflated two decisions — is this "
                 "question in the corpus's subject matter (scope), and do the retrieved chunks contain enough to "
                 "answer it (evidence) — into one yes/no, and accepted on topical overlap (ids 1, 10, 22 were judged "
                 "answerable at high similarity and produced UNSUPPORTED answers). It's replaced by two independent "
                 "stages: Stage A (`src/rag/scope_judge.py`) decides scope from the question alone, before any "
                 "retrieval; Stage B (`src/rag/evidence_judge.py`) audits retrieved passages per-proposition "
                 "(DIRECT/PARTIAL/ABSENT) rather than a single answerable boolean. Five statuses replace the old "
                 "four-state enum: `answered`, `answered_partial` (generation restricted to DIRECT passages, names "
                 "the gap), `in_scope_no_evidence` (a valid in-charter question the corpus has no evidence for — "
                 "never called out of scope), `answered_adjacent` (evidence sufficient, question outside the four "
                 "core areas), `out_of_scope` (unrelated, or adjacent + insufficient/partial). Full detail: "
                 "`RECON.md`.")
    lines.append("")

    lines.append("## 2. Answer/refusal rates by label, with 95% Wilson intervals")
    lines.append("")
    lines.append("| Label | n | New: answered | New: in_scope_no_evidence | New: out_of_scope | Old floor: answered | Prev boolean judge: answered |")
    lines.append("|---|---|---|---|---|---|---|")
    for label in ("in_scope", "adjacent", "unrelated"):
        s = stats[label]
        lines.append(
            f"| {label} | {s['n']} | {_fmt_ci(s['new_answer_rate'])} | {_fmt_ci(s['new_in_scope_no_evidence_rate'])} | "
            f"{_fmt_ci(s['new_refusal_rate'])} | {_fmt_ci(s['old_floor_answer_rate'])} | {_fmt_ci(s['prev_judge_answer_rate'])} |"
        )
    lines.append("")
    lines.append("Interval widths at n=17 and especially n=2 are wide — see §8 Limitations before reading any "
                 "single-run difference above as a real effect.")
    lines.append("")

    lines.append("## 3. Confusion matrix: Stage A scope vs. eval-set label")
    lines.append("")
    lines.append("| Stage A scope \\ eval label | in_scope | adjacent | unrelated |")
    lines.append("|---|---|---|---|")
    for scope in ("in_charter", "adjacent", "unrelated"):
        row = confusion[scope]
        lines.append(f"| {scope} | {row['in_scope']} (n={len(row['in_scope'])}) | {row['adjacent']} (n={len(row['adjacent'])}) | {row['unrelated']} (n={len(row['unrelated'])}) |")
    lines.append("")

    lines.append("## 4. Faithfulness distribution of answered cases, by label")
    lines.append("")
    lines.append("| Label | SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED |")
    lines.append("|---|---|---|---|")
    for label in ("in_scope", "adjacent", "unrelated"):
        d = faith_dist[label]
        lines.append(f"| {label} | {d['SUPPORTED']} | {d['PARTIALLY_SUPPORTED']} | {d['UNSUPPORTED']} |")
    lines.append("")

    lines.append("## 5. Per-question table (all 43)")
    lines.append("")
    lines.append("| id | label | topic | top-1 score | old status | prev judge status | new scope | new evidence verdict | new status | faithfulness |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    lines.extend(table_rows)
    lines.append("")

    lines.append("## 6. Specifically requested: ids 1, 7, 10, 18, 20, 22, 25, 33, 37")
    lines.append("")
    lines.append("| id | label | topic | top-1 score | old status | prev judge status | new scope | new evidence verdict | new status | faithfulness |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    lines.extend(callout_rows)
    lines.append("")
    lines.append("Id 20 (the classifier bug): fixed in `src/rag/query_classifier.py` before this run — see `RECON.md` §7 "
                 "for the root cause and `git log` for the fix commit. Its row above reflects the fixed classifier.")
    lines.append("")

    lines.append("## 7. Latency and token cost per stage, vs. the previous (boolean-judge) pipeline")
    lines.append("")
    lc = lat_cost

    def _fmt_mp(pair, unit="ms"):
        mean, p95 = pair
        return f"mean {mean}{unit}, p95 {p95}{unit}" if mean is not None else "n/a"

    lines.append(f"- Stage A (scope): {_fmt_mp(lc['new_scope_ms'])}")
    lines.append(f"- Retrieval: {_fmt_mp(lc['new_retrieval_ms'])}")
    lines.append(f"- Stage B (evidence): {_fmt_mp(lc['new_evidence_ms'])}")
    lines.append(f"- Generation (this run): {_fmt_mp(lc['new_generation_ms'])}")
    lines.append(f"- Generation (previous, floor-gated pipeline): {_fmt_mp(lc['old_generation_ms'])}")
    lines.append(f"- Total (this run): {_fmt_mp(lc['new_total_ms'])}")
    lines.append(f"- Mean generation tokens, this run: {lc['new_mean_input_tokens']} in / {lc['new_mean_output_tokens']} out")
    lines.append(f"- Mean generation tokens, previous pipeline: {lc['old_mean_input_tokens']} in / {lc['old_mean_output_tokens']} out")
    lines.append("")
    lines.append(f"Models actually called (read from each response's own `.model` field, not config defaults): "
                 f"scope — {', '.join(models['scope_models']) or 'none'}; evidence — {', '.join(models['evidence_models']) or 'none'}; "
                 f"generation — {', '.join(models['generation_models']) or 'none'}. Faithfulness grader model: "
                 f"see `data/eval_v2_judge_twostage_v2_5k_faithfulness.json`'s `grader_model` field.")
    lines.append("")

    lines.append("## 8. Limitations")
    lines.append("")
    lines.append("- **n=43 is small, and n=2 for the unrelated group is not enough to estimate a rate at all** — "
                 "read the Wilson intervals in §2 literally: most of them overlap across old/prev/new, meaning most "
                 "apparent differences between pipelines in this report are not statistically distinguishable from "
                 "noise at this sample size. Treat directional agreement across several ids as more informative "
                 "than any single rate.")
    lines.append("- **This eval set was used to diagnose the exact problems this rebuild fixes** (ids 1, 7, 10, 18, "
                 "20, 22, 25, 33, 37 were named in the task specification itself, based on the previous pipeline's "
                 "known failures on them). Numbers in this report are optimistic for that reason — they measure "
                 "whether the fix works on the cases known to motivate it, not on held-out questions the rebuild "
                 "was never informed by. Confirming this on a genuinely new question set is the natural next step, "
                 "not done here.")
    lines.append("- **Faithfulness-grader independence**: the grader uses a different model (Sonnet, via "
                 "`src.rag.chain.get_model()`) and independently-worded rubric from Stage A/B (both Haiku, via "
                 "`src.rag.judge.get_judge_model()`) — verified by a unit test asserting zero shared distinctive "
                 "vocabulary between the two prompts. This makes agreement between the grader and Stage B more "
                 "informative than if they shared a model, but it does not make the grader a validated ground truth "
                 "instrument — it is itself an LLM judgment, unreplicated by a second grading pass or a human rater "
                 "beyond the `eval/hand_grading.csv` spot-check.")
    lines.append("- **No prompt iteration**: per the task's explicit rule, neither judge prompt was changed based on "
                 "this run's results — this run happened exactly once. Two real bugs were fixed before this run "
                 "(the `direct`/`partial` chunk-ref parsing accepting both int and \"C{n}\" string forms, and the "
                 "id-20 classifier misroute) — both caught by direct testing before the eval, not by tuning against "
                 "its output.")
    lines.append("- **Single run, no variance estimate** beyond the Wilson intervals on the rates themselves — "
                 "every judge/evidence/generation/grading call is a real, non-deterministic LLM call (this SDK build "
                 "has no `temperature` parameter to reduce variance with — see `src/rag/scope_judge.py`'s note). "
                 "Individual ids could grade differently on a re-run.")
    lines.append("")

    lines.append("## 9. Consistency assertions")
    lines.append("")
    lines.append("```")
    lines.extend(assertion_log)
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
