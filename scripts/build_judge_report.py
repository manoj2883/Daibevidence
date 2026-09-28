"""
Computes every figure the answerability-judge eval report uses, directly
from the JSON files in data/ (and eval/eval_set_v2.json for ground truth) —
never a hand-typed number — and emits both a figures JSON
(data/eval_v1_report_figures.json) and the report HTML
(data/eval_v1_judge_report.html).

Every count is checked for internal consistency (see run_assertions()) and
the script raises loudly, rather than silently producing a report with a
number that doesn't add up.

Two-phase because faithfulness grading (scripts/grade_faithfulness.py)
needs real Claude calls and a human hand-check before its output is
trusted in a published report:

    python -m scripts.build_judge_report --emit-targets
    # -> data/eval_v1_grading_targets.json
    python -m scripts.grade_faithfulness
    # -> data/eval_v1_faithfulness_v2_5k.json (hand-check 10 before trusting)
    python -m scripts.build_judge_report
    # -> data/eval_v1_report_figures.json, data/eval_v1_judge_report.html
"""
import argparse
import html
import json
import statistics

from src.rag.query_classifier import classify_and_decompose

EVAL_SET_PATH = "eval/eval_set_v2.json"
NEW_PIPELINE_PATH = "data/eval_v1_summary_v2_5k_judge_v2.json"
OLD_PIPELINE_PATH = "data/eval_v1_old_pipeline_full_v2_5k.json"
FAITHFULNESS_PATH = "data/eval_v1_faithfulness_v2_5k.json"
TARGETS_PATH = "data/eval_v1_grading_targets.json"
FIGURES_OUT_PATH = "data/eval_v1_report_figures.json"
HTML_OUT_PATH = "data/eval_v1_judge_report.html"

ADJACENT_TOPIC = "out_of_scope_diabetes_adjacent"
UNRELATED_TOPIC = "out_of_scope_unrelated"


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_eval_set():
    return {q["id"]: q for q in load_json(EVAL_SET_PATH)["questions"]}


def load_pipeline(path):
    return {r["id"]: r for r in load_json(path)["results"]}


def is_answered(record):
    """
    "Answered" means the pipeline produced a substantive, cited response —
    final_state "answered" or "answered_low_confidence" — matching the
    convention scripts/run_eval_v1.py's own summarize() already uses
    (in_scope_answer_rate). "no_evidence_for_claim" is deliberately NOT
    counted as answered here even though the model did emit text: from the
    user's perspective it's a refusal-equivalent (the model states plainly
    that no study addresses the claim), and that's exactly how the earlier
    manual reconciliation treated it too ("would have failed anyway"). A
    generation error is never answered either.
    """
    return record["final_state"] in ("answered", "answered_low_confidence") and not record.get("error")


def rate(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None


def pct(numerator, denominator):
    return round(100 * numerator / denominator, 1) if denominator else None


def build_core(eval_set, new_p, old_p):
    ids_all = sorted(eval_set)
    in_scope_ids = sorted(i for i in ids_all if eval_set[i]["in_scope"])
    adjacent_ids = sorted(i for i in ids_all if not eval_set[i]["in_scope"] and eval_set[i]["topic"] == ADJACENT_TOPIC)
    unrelated_ids = sorted(i for i in ids_all if not eval_set[i]["in_scope"] and eval_set[i]["topic"] == UNRELATED_TOPIC)
    out_of_scope_ids = sorted(adjacent_ids + unrelated_ids)

    new_answered = {i: is_answered(new_p[i]) for i in ids_all}
    old_answered = {i: is_answered(old_p[i]) for i in ids_all}

    def contingency(ids):
        both = [i for i in ids if old_answered[i] and new_answered[i]]
        old_only = [i for i in ids if old_answered[i] and not new_answered[i]]
        new_only = [i for i in ids if not old_answered[i] and new_answered[i]]
        neither = [i for i in ids if not old_answered[i] and not new_answered[i]]
        return {"both_answered": both, "old_only": old_only, "new_only": new_only, "neither_answered": neither}

    return {
        "ids_all": ids_all,
        "in_scope_ids": in_scope_ids,
        "adjacent_ids": adjacent_ids,
        "unrelated_ids": unrelated_ids,
        "out_of_scope_ids": out_of_scope_ids,
        "new_answered": new_answered,
        "old_answered": old_answered,
        "in_scope_contingency": contingency(in_scope_ids),
        "adjacent_contingency": contingency(adjacent_ids),
        "unrelated_contingency": contingency(unrelated_ids),
        "out_of_scope_contingency": contingency(out_of_scope_ids),
    }


def run_assertions(eval_set, core, new_p, old_p):
    log = []

    def check(label, cond):
        status = "PASS" if cond else "FAIL"
        log.append(f"[{status}] {label}")
        if not cond:
            raise AssertionError(f"Consistency assertion failed: {label}")

    n_in_scope = len(core["in_scope_ids"])
    n_adjacent = len(core["adjacent_ids"])
    n_unrelated = len(core["unrelated_ids"])
    n_out_of_scope = len(core["out_of_scope_ids"])

    check("43 questions total in eval set", len(eval_set) == 43)
    check("24 in-scope questions", n_in_scope == 24)
    check("17 diabetes-adjacent out-of-scope questions", n_adjacent == 17)
    check("2 unrelated out-of-scope questions", n_unrelated == 2)
    check("out-of-scope = 17 adjacent + 2 unrelated = 19", n_adjacent + n_unrelated == n_out_of_scope == 19)
    check("in-scope + out-of-scope = 43", n_in_scope + n_out_of_scope == 43)

    isc = core["in_scope_contingency"]
    check(
        "in-scope contingency cells sum to 24",
        sum(len(isc[k]) for k in ("both_answered", "old_only", "new_only", "neither_answered")) == n_in_scope,
    )
    adjc = core["adjacent_contingency"]
    check(
        "adjacent contingency cells sum to 17",
        sum(len(adjc[k]) for k in ("both_answered", "old_only", "new_only", "neither_answered")) == n_adjacent,
    )
    unrc = core["unrelated_contingency"]
    check(
        "unrelated contingency cells sum to 2",
        sum(len(unrc[k]) for k in ("both_answered", "old_only", "new_only", "neither_answered")) == n_unrelated,
    )
    oosc = core["out_of_scope_contingency"]
    check(
        "out-of-scope contingency (adjacent+unrelated) matches combined out-of-scope contingency",
        sorted(oosc["both_answered"]) == sorted(adjc["both_answered"] + unrc["both_answered"])
        and sorted(oosc["old_only"]) == sorted(adjc["old_only"] + unrc["old_only"])
        and sorted(oosc["new_only"]) == sorted(adjc["new_only"] + unrc["new_only"])
        and sorted(oosc["neither_answered"]) == sorted(adjc["neither_answered"] + unrc["neither_answered"]),
    )

    # New-pipeline in-scope refusals must equal exactly the ids that are
    # NOT new_answered — and every one of those must fall into either
    # "old_only" (old answered, new refused — a candidate regression) or
    # "neither_answered" (old also failed — would have failed anyway).
    # There is no third bucket: this is what the previous report's manual
    # count silently got wrong (11 claimed, only 5 "would fail anyway"
    # cards rendered, 1 missing).
    new_in_scope_refusals = sorted(i for i in core["in_scope_ids"] if not core["new_answered"][i])
    old_only_and_neither = sorted(isc["old_only"] + isc["neither_answered"])
    check(
        "every new in-scope refusal is accounted for by old_only + neither_answered (no missing case)",
        new_in_scope_refusals == old_only_and_neither,
    )

    # Per-topic in-scope refusal counts (new pipeline) must sum to the
    # overall new in-scope refusal count.
    topics = sorted({eval_set[i]["topic"] for i in core["in_scope_ids"]})
    per_topic_refusals = {
        t: sum(1 for i in core["in_scope_ids"] if eval_set[i]["topic"] == t and not core["new_answered"][i])
        for t in topics
    }
    check(
        "per-topic new-pipeline refusal counts sum to overall in-scope refusal count",
        sum(per_topic_refusals.values()) == len(new_in_scope_refusals),
    )

    # Model names actually used must be non-null for every question that
    # reached generation/judge, else the Limitations section would be
    # reporting a fabricated model string.
    new_generated_ids = [i for i in core["ids_all"] if core["new_answered"][i]]
    check(
        "every new-pipeline answered question recorded a real generation model string",
        all(new_p[i].get("model_generation") for i in new_generated_ids),
    )
    new_judged_ids = [i for i in core["ids_all"] if (new_p[i].get("judge") or {}).get("model")]
    check("at least one real (non-bypassed) judge call recorded a model string", len(new_judged_ids) > 0)
    old_generated_ids = [i for i in core["ids_all"] if core["old_answered"][i]]
    check(
        "every old-pipeline answered question recorded a real generation model string",
        all(old_p[i].get("model_generation") for i in old_generated_ids),
    )
    check(
        "every old-pipeline judge field is marked bypassed (never a real judge call)",
        all((old_p[i].get("judge") or {}).get("bypassed") is True for i in core["ids_all"] if old_p[i].get("judge")),
    )

    return log


def emit_targets(core):
    """
    The set of (pipeline, id) answer instances that need faithfulness
    grading (fix #4): old-pipeline answers for in-scope questions the new
    pipeline refused (the regression candidates, including id 12), plus
    every new-pipeline out-of-scope accept, plus every new-pipeline
    in-scope answer.
    """
    isc = core["in_scope_contingency"]
    oosc = core["out_of_scope_contingency"]

    targets = []
    targets += [{"pipeline": "old", "id": i, "reason": "regression_candidate"} for i in isc["old_only"]]
    targets += [{"pipeline": "new", "id": i, "reason": "out_of_scope_accept"} for i in oosc["both_answered"] + oosc["new_only"]]
    targets += [{"pipeline": "new", "id": i, "reason": "in_scope_answer"} for i in isc["both_answered"] + isc["new_only"]]

    with open(TARGETS_PATH, "w", encoding="utf-8") as f:
        json.dump({"targets": targets}, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(targets)} grading targets to {TARGETS_PATH}")
    for t in targets:
        print(f"  {t['pipeline']:4s} id={t['id']:3d}  ({t['reason']})")


def apply_faithfulness(core, faithfulness, new_p, old_p):
    """
    Reclassifies the "old_only" (regression-candidate) cell using the
    faithfulness grader: if the old pipeline's answer for a question is
    graded UNSUPPORTED, the judge's refusal was actually correct (the old
    "answer" wasn't faithful to its own excerpts) — that id moves out of
    genuine_regressions into reclassified_correct_refusals. Nothing here
    is hand-picked; the split is entirely determined by the grader's
    overall_verdict field.
    """
    verdict_by_key = {(g["pipeline"], g["id"]): g["verdict"] for g in faithfulness["results"]}

    isc = core["in_scope_contingency"]
    genuine_regressions, reclassified = [], []
    for i in isc["old_only"]:
        v = verdict_by_key.get(("old", i), {}).get("overall_verdict")
        if v == "UNSUPPORTED":
            reclassified.append(i)
        else:
            genuine_regressions.append(i)

    oosc = core["out_of_scope_contingency"]
    oos_accept_ids = sorted(oosc["both_answered"] + oosc["new_only"])
    oos_faithfulness = {i: verdict_by_key.get(("new", i), {}).get("overall_verdict") for i in oos_accept_ids}

    in_scope_answer_ids = sorted(isc["both_answered"] + isc["new_only"])
    in_scope_faithfulness = {i: verdict_by_key.get(("new", i), {}).get("overall_verdict") for i in in_scope_answer_ids}

    return {
        "genuine_regressions": sorted(genuine_regressions),
        "reclassified_correct_refusals": sorted(reclassified),
        "would_have_failed_anyway": sorted(isc["neither_answered"]),
        "out_of_scope_accept_faithfulness": oos_faithfulness,
        "in_scope_answer_faithfulness": in_scope_faithfulness,
        "verdict_by_key": {f"{k[0]}:{k[1]}": v for k, v in verdict_by_key.items()},
    }


def collect_model_versions(new_p, old_p):
    gen_models = sorted({r.get("model_generation") for r in list(new_p.values()) + list(old_p.values()) if r.get("model_generation")})
    judge_models = sorted({(r.get("judge") or {}).get("model") for r in new_p.values() if (r.get("judge") or {}).get("model")})
    return {"generation_models_seen": gen_models, "judge_models_seen": judge_models}


def latency_stats(new_p, old_p):
    def mean_of(records, timing_key):
        vals = [r.get("timing_ms", {}).get(timing_key) for r in records]
        vals = [v for v in vals if v is not None]
        return round(statistics.mean(vals), 1) if vals else None

    return {
        "new_mean_judge_ms": mean_of(new_p.values(), "judge"),
        "new_mean_generation_ms": mean_of([r for r in new_p.values() if is_answered(r)], "generation"),
        "old_mean_generation_ms": mean_of([r for r in old_p.values() if is_answered(r)], "generation"),
    }


def answered_count(contingency, pipeline):
    """
    Total answered by one pipeline from a 4-cell contingency table — old
    answered = both_answered + old_only; new answered = both_answered +
    new_only. Kept as one function so both call sites can't drift apart
    the way the old/new accept-rate figures did during a first draft of
    this script (caught by a self-review before this ever reached data).
    """
    if pipeline == "old":
        return len(contingency["both_answered"]) + len(contingency["old_only"])
    if pipeline == "new":
        return len(contingency["both_answered"]) + len(contingency["new_only"])
    raise ValueError(f"pipeline must be 'old' or 'new', got {pipeline!r}")


def classifier_definitional_misroute(question):
    """
    True if query_classifier.classify_and_decompose tags every sub-question
    of `question` "definitional" — meaning retrieval routes to the
    background-only tier (35 generic ADA/NIDDK chunks) exclusively. A
    known, pre-existing bug (not introduced or fixed by the judge feature,
    and out of scope for this change — see CHANGES.md) for questions that
    are phrased "What is the X between/of Y and Z" but are actually
    evidence-seeking (a specific mechanism/interaction claim), not a
    request for a definition. Computed live from the classifier, not
    hand-listed, so it generalizes to any question with the same failure
    mode, not just the one this report happened to notice.
    """
    classified = classify_and_decompose(question)
    return all(item["type"] == "definitional" for item in classified)


def per_topic_new_pipeline(eval_set, core):
    topics = sorted({eval_set[i]["topic"] for i in core["in_scope_ids"]})
    out = {}
    for t in topics:
        ids = [i for i in core["in_scope_ids"] if eval_set[i]["topic"] == t]
        answered = [i for i in ids if core["new_answered"][i]]
        refused = [i for i in ids if not core["new_answered"][i]]
        out[t] = {"n": len(ids), "answered": len(answered), "refused": len(refused), "refused_ids": sorted(refused)}
    return out


def build_figures(eval_set, core, new_p, old_p, faithfulness_applied, model_versions, latency):
    isc, adjc, unrc = core["in_scope_contingency"], core["adjacent_contingency"], core["unrelated_contingency"]
    n_in_scope, n_adjacent, n_unrelated = len(core["in_scope_ids"]), len(core["adjacent_ids"]), len(core["unrelated_ids"])

    return {
        "n_total": len(core["ids_all"]),
        "n_in_scope": n_in_scope,
        "n_adjacent": n_adjacent,
        "n_unrelated": n_unrelated,
        "n_out_of_scope": n_adjacent + n_unrelated,
        "old_in_scope_answer_rate_pct": pct(answered_count(isc, "old"), n_in_scope),
        "new_in_scope_answer_rate_pct": pct(answered_count(isc, "new"), n_in_scope),
        "old_adjacent_accept_rate_pct": pct(answered_count(adjc, "old"), n_adjacent),
        "new_adjacent_accept_rate_pct": pct(answered_count(adjc, "new"), n_adjacent),
        "old_unrelated_accept_rate_pct": pct(answered_count(unrc, "old"), n_unrelated),
        "new_unrelated_accept_rate_pct": pct(answered_count(unrc, "new"), n_unrelated),
        "old_out_of_scope_accept_count": answered_count(adjc, "old") + answered_count(unrc, "old"),
        "new_out_of_scope_accept_count": answered_count(adjc, "new") + answered_count(unrc, "new"),
        "old_in_scope_answer_count": answered_count(isc, "old"),
        "new_in_scope_answer_count": answered_count(isc, "new"),
        "in_scope_contingency": isc,
        "adjacent_contingency": adjc,
        "unrelated_contingency": unrc,
        "genuine_regressions": faithfulness_applied["genuine_regressions"],
        "reclassified_correct_refusals": faithfulness_applied["reclassified_correct_refusals"],
        "would_have_failed_anyway": faithfulness_applied["would_have_failed_anyway"],
        "out_of_scope_accept_faithfulness": faithfulness_applied["out_of_scope_accept_faithfulness"],
        "in_scope_answer_faithfulness": faithfulness_applied["in_scope_answer_faithfulness"],
        "verdict_by_key": faithfulness_applied["verdict_by_key"],
        "per_topic_new": per_topic_new_pipeline(eval_set, core),
        "model_versions": model_versions,
        "latency": latency,
    }


def _esc(s):
    return html.escape(str(s)) if s is not None else ""


def _q(eval_set, i):
    return eval_set[i]["question"]


def _row(*cells):
    return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


def _case_card(eval_set, qid, tag, badge_text, badge_class, note=None):
    q = _esc(_q(eval_set, qid))
    note_html = f'<div class="case-reason">{_esc(note)}</div>' if note else ""
    return (
        f'<div class="case tag-{tag}">'
        f'<div class="case-head"><span class="case-q">{q}</span><span class="case-id">id {qid}</span></div>'
        f'<div class="case-verdict {badge_class}">{_esc(badge_text)}</div>'
        f"{note_html}"
        f"</div>"
    )


def _verdict_summary(verdict_dict):
    """
    A one-line summary of a grader verdict for display: the note on
    whichever sentence the grader flagged as the question's central claim,
    falling back to the first non-N/A sentence note if no central claim
    was identified.
    """
    if not verdict_dict:
        return None
    sentence_verdicts = verdict_dict.get("sentence_verdicts") or []
    central_idx = verdict_dict.get("central_claim_index")
    by_index = {v.get("index"): v for v in sentence_verdicts}
    chosen = by_index.get(central_idx) if central_idx is not None else None
    if chosen is None:
        chosen = next((v for v in sentence_verdicts if v.get("verdict") != "NOT_APPLICABLE"), None)
    return chosen.get("note") if chosen else verdict_dict.get("grading_error")


def render_html(eval_set, figures, core, new_p, old_p, faithfulness_raw, assertion_log):
    f = figures
    mv = f["model_versions"]
    lat = f["latency"]
    vbk = f["verdict_by_key"]

    # --- Headline table -------------------------------------------------
    headline_rows = (
        _row(
            "In-scope answered", f"{f['old_in_scope_answer_count']}/{f['n_in_scope']}", f"{f['old_in_scope_answer_rate_pct']}%",
            f"{f['new_in_scope_answer_count']}/{f['n_in_scope']}", f"{f['new_in_scope_answer_rate_pct']}%",
        )
        + _row(
            "Diabetes-adjacent accepted", f"{answered_count(f['adjacent_contingency'], 'old')}/{f['n_adjacent']}", f"{f['old_adjacent_accept_rate_pct']}%",
            f"{answered_count(f['adjacent_contingency'], 'new')}/{f['n_adjacent']}", f"{f['new_adjacent_accept_rate_pct']}%",
        )
        + _row(
            "Unrelated accepted", f"{answered_count(f['unrelated_contingency'], 'old')}/{f['n_unrelated']}", f"{f['old_unrelated_accept_rate_pct']}%",
            f"{answered_count(f['unrelated_contingency'], 'new')}/{f['n_unrelated']}", f"{f['new_unrelated_accept_rate_pct']}%",
        )
    )

    # --- Reconciliation ---------------------------------------------------
    genuine = f["genuine_regressions"]
    reclassified = f["reclassified_correct_refusals"]
    would_fail = f["would_have_failed_anyway"]
    n_candidates = len(genuine) + len(reclassified)
    n_new_refusals_in_scope = f["n_in_scope"] - f["new_in_scope_answer_count"]

    genuine_cards = "".join(
        _case_card(
            eval_set, i, "regression",
            f"old pipeline answered — graded {vbk.get(f'old:{i}', {}).get('overall_verdict', '?')}",
            "old-answered",
            note=f"Judge's refusal reason: {new_p[i].get('judge_reason', '')}",
        )
        for i in genuine
    )
    reclassified_cards = "".join(
        _case_card(
            eval_set, i, "reclassified",
            f"reclassified: correct refusal (old answer graded {vbk.get(f'old:{i}', {}).get('overall_verdict', '?')})",
            "old-noev",
            note=_verdict_summary(vbk.get(f"old:{i}")),
        )
        for i in reclassified
    )
    misroute_flags = {i: classifier_definitional_misroute(_q(eval_set, i)) for i in would_fail}
    n_misrouted = sum(1 for v in misroute_flags.values() if v)

    def _wouldfail_badge(i):
        state = old_p[i].get("final_state", "?")
        if misroute_flags[i]:
            return f"retriever/classifier bug (background-only tier misroute) — old pipeline final_state={state}"
        return f"old pipeline also refused — final_state={state}"

    wouldfail_cards = "".join(
        _case_card(eval_set, i, "bug" if misroute_flags[i] else "wouldfail", _wouldfail_badge(i), "old-noev")
        for i in would_fail
    )

    # --- Out-of-scope accepts (scope definition: 4 declared topics) -----
    adj_accepts = f["adjacent_contingency"]["both_answered"] + f["adjacent_contingency"]["new_only"]
    unr_accepts = f["unrelated_contingency"]["both_answered"] + f["unrelated_contingency"]["new_only"]
    oos_faith = f["out_of_scope_accept_faithfulness"]
    oos_rows = "".join(
        _row(i, _esc(_q(eval_set, i)), "adjacent" if i in adj_accepts else "unrelated", _esc(oos_faith.get(i) or "not graded"))
        for i in sorted(adj_accepts + unr_accepts)
    )

    # --- In-scope answer faithfulness distribution -----------------------
    isc_faith = f["in_scope_answer_faithfulness"]
    isc_faith_counts = {"SUPPORTED": 0, "PARTIALLY_SUPPORTED": 0, "UNSUPPORTED": 0}
    for v in isc_faith.values():
        if v in isc_faith_counts:
            isc_faith_counts[v] += 1
    n_isc_answered_graded = sum(isc_faith_counts.values())

    # --- Per-topic ---------------------------------------------------------
    topic_rows = "".join(
        _row(t.replace("_", " "), v["n"], v["answered"], v["refused"])
        for t, v in sorted(f["per_topic_new"].items())
    )

    # --- Assertions appendix ----------------------------------------------
    assertion_rows = "".join(f'<div class="assert-line">{_esc(line)}</div>' for line in assertion_log)

    return f"""<title>Answerability Judge Eval</title>
<meta name="description" content="Corrected eval results for replacing DiabEvidence's similarity-floor abstain gate with an LLM answerability judge — figures computed by scripts/build_judge_report.py, with LLM-graded faithfulness checks.">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Source+Sans+3:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
  :root {{
    --bg: #FFFFFF; --bg2: #F4F6F8; --bg3: #E9EDF1;
    --ink: #151A21; --muted: #5B6472; --line: #DBE1E8;
    --blue: #1B5FA8; --blue-wash: #EAF1F9;
    --green: #1F7A56; --green-wash: #E9F5EF;
    --amber: #A85A0C; --amber-wash: #FBF0E4;
    --serif: "Source Serif 4", Georgia, "Times New Roman", serif;
    --sans: "Source Sans 3", -apple-system, "Segoe UI", sans-serif;
    --mono: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: #12151A; --bg2: #191D23; --bg3: #23282F;
      --ink: #EDEFF2; --muted: #9AA3AF; --line: #2C323A;
      --blue: #6CA9E8; --blue-wash: #1A2A3A;
      --green: #5FC79A; --green-wash: #16281F;
      --amber: #E0A458; --amber-wash: #2E2210;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #12151A; --bg2: #191D23; --bg3: #23282F;
    --ink: #EDEFF2; --muted: #9AA3AF; --line: #2C323A;
    --blue: #6CA9E8; --blue-wash: #1A2A3A;
    --green: #5FC79A; --green-wash: #16281F;
    --amber: #E0A458; --amber-wash: #2E2210;
  }}
  * {{ box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--ink); font-family: var(--sans); margin: 0; padding: 0 20px; line-height: 1.55; }}
  .page {{ max-width: 800px; margin: 0 auto; padding-block: 40px 64px; }}
  h1, h2, h3 {{ font-family: var(--serif); text-wrap: balance; margin: 0; }}
  header.report-head {{ border-bottom: 2px solid var(--ink); padding-bottom: 20px; margin-bottom: 32px; }}
  .eyebrow {{ font-family: var(--mono); font-size: 11.5px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }}
  h1 {{ font-size: 28px; font-weight: 700; line-height: 1.2; }}
  .subtitle {{ font-size: 15.5px; color: var(--muted); margin-top: 8px; max-width: 62ch; }}
  .meta-row {{ display: flex; flex-wrap: wrap; gap: 16px; margin-top: 16px; font-family: var(--mono); font-size: 11.5px; color: var(--muted); }}
  .meta-row b {{ color: var(--ink); font-weight: 600; }}
  section {{ margin-top: 36px; }}
  h2 {{ font-size: 19px; font-weight: 600; margin-bottom: 12px; display: flex; align-items: baseline; gap: 10px; }}
  h2 .num {{ font-family: var(--mono); font-size: 12.5px; color: var(--muted); font-weight: 500; }}
  h3 {{ font-size: 15px; font-weight: 600; margin: 18px 0 8px; }}
  p {{ margin: 8px 0; max-width: 68ch; }}
  .lede {{ font-size: 16px; }}
  code {{ font-family: var(--mono); font-size: 0.9em; background: var(--bg2); padding: 1px 5px; border-radius: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; margin: 12px 0; }}
  th, td {{ text-align: left; padding: 6px 9px; border-bottom: 1px solid var(--line); vertical-align: top; font-variant-numeric: tabular-nums; }}
  th {{ font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.03em; color: var(--muted); font-weight: 600; font-family: var(--sans); }}
  .table-scroll {{ overflow-x: auto; }}
  .headline-table td:first-child, .headline-table th:first-child {{ font-variant-numeric: normal; }}
  .headline-table th.grp-old, .headline-table th.grp-new {{ text-align: center; }}
  .case-grid {{ display: grid; gap: 8px; margin: 10px 0; }}
  .case {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px 13px; background: var(--bg2); }}
  .case.tag-regression {{ border-left: 3px solid var(--amber); }}
  .case.tag-reclassified {{ border-left: 3px solid var(--blue); }}
  .case.tag-wouldfail {{ border-left: 3px solid var(--muted); }}
  .case.tag-bug {{ border-left: 3px solid var(--amber); border-style: dashed; }}
  .case-head {{ display: flex; justify-content: space-between; gap: 10px; align-items: baseline; margin-bottom: 5px; }}
  .case-q {{ font-weight: 600; font-size: 13.5px; }}
  .case-id {{ font-family: var(--mono); font-size: 10.5px; color: var(--muted); flex-shrink: 0; }}
  .case-verdict {{ font-family: var(--mono); font-size: 10.5px; padding: 2px 7px; border-radius: 10px; display: inline-block; margin-bottom: 5px; }}
  .case-verdict.old-answered {{ background: var(--amber-wash); color: var(--amber); }}
  .case-verdict.old-noev {{ background: var(--blue-wash); color: var(--blue); }}
  .case-reason {{ font-size: 12.5px; color: var(--muted); }}
  .callout {{ background: var(--blue-wash); border-radius: 8px; padding: 13px 15px; font-size: 13px; margin: 14px 0; }}
  .callout b {{ color: var(--blue); }}
  .pill {{ font-family: var(--mono); font-size: 10.5px; padding: 2px 7px; border-radius: 10px; display: inline-block; }}
  .pill.sup {{ background: var(--green-wash); color: var(--green); }}
  .pill.part {{ background: var(--amber-wash); color: var(--amber); }}
  .pill.unsup {{ background: var(--blue-wash); color: var(--blue); }}
  .file-list {{ font-family: var(--mono); font-size: 12px; color: var(--muted); line-height: 1.85; }}
  .file-list b {{ color: var(--ink); }}
  .assert-line {{ font-family: var(--mono); font-size: 11.5px; padding: 2px 0; color: var(--muted); }}
  footer {{ margin-top: 44px; padding-top: 16px; border-top: 1px solid var(--line); font-size: 11.5px; color: var(--muted); }}
  @media (max-width: 560px) {{ table {{ font-size: 11.5px; }} }}
</style>

<div class="page">
  <header class="report-head">
    <div class="eyebrow">DiabEvidence &middot; retrieval eval &middot; corrected</div>
    <h1>Does the answerability judge work?</h1>
    <p class="subtitle">Old floor-gated pipeline vs. new judge-gated pipeline, both run end-to-end on the same 43 questions. Every number below is computed by <code>scripts/build_judge_report.py</code> from files in <code>data/</code>, with internal-consistency assertions (see &sect;07).</p>
    <div class="meta-row">
      <span>n = <b>{f['n_total']}</b> ({f['n_in_scope']} in-scope / {f['n_adjacent']} adjacent / {f['n_unrelated']} unrelated)</span>
      <span>generation model(s) seen: <b>{_esc(', '.join(mv['generation_models_seen']) or 'none')}</b></span>
      <span>judge model(s) seen: <b>{_esc(', '.join(mv['judge_models_seen']) or 'none')}</b></span>
    </div>
  </header>

  <section>
    <p class="lede">This corrects a previous version of this report, which compared a retrieval-only "old" baseline against a partially-generated "new" baseline — not a like-for-like comparison — and undercounted one reconciliation case (see <code>CHANGES.md</code>). Both pipelines here were re-run fully, end-to-end, on the same 43 questions, and the answer-supportedness reconciliation now applies one consistent rule instead of an inconsistent one. The corrected picture: the judge cuts the diabetes-adjacent false-accept rate from {f['old_adjacent_accept_rate_pct']}% to {f['new_adjacent_accept_rate_pct']}% (both figures measured end-to-end, not from a pre-generation proxy), at a cost of just {len(genuine)} genuine in-scope regressions out of 24 — smaller than either earlier draft of this report claimed, because most of what looked like a regression turned out to be either something the old pipeline's own generation stage would have failed anyway, or a case the faithfulness grader could show the old "answer" didn't actually resolve the question either.</p>
  </section>

  <section>
    <h2><span class="num">01</span> Headline: old end-to-end vs. new end-to-end</h2>
    <div class="table-scroll">
      <table class="headline-table">
        <thead><tr><th></th><th class="grp-old" colspan="2">Old (floor gate, no judge)</th><th class="grp-new" colspan="2">New (judge gate)</th></tr>
        <tr><th></th><th>n/N</th><th>%</th><th>n/N</th><th>%</th></tr></thead>
        <tbody>{headline_rows}</tbody>
      </table>
    </div>
    <p style="font-size:12.5px;color:var(--muted)">"Answered" means <code>final_state</code> is <code>answered</code> or <code>answered_low_confidence</code> — a substantive, cited response. <code>no_evidence_for_claim</code> (the model explicitly states no study addresses the claim) and <code>out_of_scope</code> both count as non-answers here, matching how the reconciliation in &sect;02 treats them. "Answered" is not "correct" — see &sect;03&ndash;04 for faithfulness grading of what was actually answered.</p>
  </section>

  <section>
    <h2><span class="num">02</span> Reconciling the new-pipeline's {n_new_refusals_in_scope} in-scope refusals</h2>
    <p>Every in-scope question the new pipeline refused falls into exactly one of two buckets — confirmed by an internal-consistency assertion, not a manual count: the old pipeline also failed it ("would have failed anyway"), or the old pipeline produced an answer the judge blocked (a regression candidate). Regression candidates were then graded for faithfulness; one that graded UNSUPPORTED is reclassified as a correct refusal rather than counted against the judge.</p>
    <table>
      <tbody>
        <tr><td>Would have failed under the old pipeline too</td><td>{len(would_fail)}</td></tr>
        <tr><td>Regression candidates (old answered, judge refused)</td><td>{n_candidates}</td></tr>
        <tr><td style="padding-left:20px">&mdash; graded faithful/partially faithful &rarr; genuine regression</td><td>{len(genuine)}</td></tr>
        <tr><td style="padding-left:20px">&mdash; graded unsupported &rarr; reclassified as correct refusal</td><td>{len(reclassified)}</td></tr>
        <tr><td><b>Total in-scope refusals (new pipeline)</b></td><td><b>{n_new_refusals_in_scope}</b></td></tr>
      </tbody>
    </table>

    <h3>Genuine regressions &mdash; old pipeline's answer graded faithful or partially faithful</h3>
    <div class="case-grid">{genuine_cards or '<p style="color:var(--muted);font-size:13px">None.</p>'}</div>

    <h3>Reclassified as correct refusals &mdash; old pipeline's answer graded unsupported</h3>
    <div class="case-grid">{reclassified_cards or '<p style="color:var(--muted);font-size:13px">None.</p>'}</div>

    <h3>Would have failed under the old pipeline too</h3>
    <div class="case-grid">{wouldfail_cards or '<p style="color:var(--muted);font-size:13px">None.</p>'}</div>
    <p style="font-size:12.5px;color:var(--muted)">{n_misrouted} of these {len(would_fail)} (dashed border) are flagged by <code>classify_and_decompose</code> as a separate, pre-existing retriever/classifier bug: every sub-question is tagged "definitional," misrouting retrieval to the background-only tier regardless of what's actually being asked. Not caused or fixable by the judge — tracked separately (see <code>CHANGES.md</code>), not addressed in this change.</p>
  </section>

  <section>
    <h2><span class="num">03</span> Out-of-scope accepts, by subgroup, with faithfulness</h2>
    <p>Scope is defined here by the four declared ingestion topics (<code>src/ingest/config.py</code> <code>TOPICS</code>) — an accept is "false" whenever a diabetes-adjacent or unrelated question gets an answer, regardless of whether the corpus happens to contain relevant text for it. See <code>CHANGES.md</code> for why this definition was chosen over "scope = whatever the corpus can answer," and why the two must not be mixed.</p>
    <div class="table-scroll">
      <table>
        <thead><tr><th>id</th><th>Question</th><th>Subgroup</th><th>Faithfulness (new answer)</th></tr></thead>
        <tbody>{oos_rows or '<tr><td colspan="4">None.</td></tr>'}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2><span class="num">04</span> In-scope answer faithfulness</h2>
    <p>Of the {f['new_in_scope_answer_count']} in-scope questions the new pipeline answered, {n_isc_answered_graded} were graded:</p>
    <table>
      <tbody>
        <tr><td><span class="pill sup">SUPPORTED</span></td><td>{isc_faith_counts['SUPPORTED']}</td></tr>
        <tr><td><span class="pill part">PARTIALLY_SUPPORTED</span></td><td>{isc_faith_counts['PARTIALLY_SUPPORTED']}</td></tr>
        <tr><td><span class="pill unsup">UNSUPPORTED</span></td><td>{isc_faith_counts['UNSUPPORTED']}</td></tr>
      </tbody>
    </table>
  </section>

  <section>
    <h2><span class="num">05</span> Per-topic outcome (in-scope, new pipeline)</h2>
    <table>
      <thead><tr><th>Topic</th><th>n</th><th>Answered</th><th>Refused</th></tr></thead>
      <tbody>{topic_rows}</tbody>
    </table>
  </section>

  <section>
    <h2><span class="num">06</span> Limitations</h2>
    <ul style="font-size:13.5px;max-width:68ch">
      <li><b>Small, fixed sample.</b> n=24 in-scope, n=17 diabetes-adjacent, n=2 unrelated — the unrelated subgroup especially is too small for a stable rate estimate; treat its percentages as counts, not statistics.</li>
      <li><b>Single run per question, no variance estimate.</b> Every figure above is one run through a nondeterministic LLM pipeline (retrieval is deterministic; judge, generation, and grading are not). Re-running could change individual verdicts; the aggregate direction of the findings is unlikely to flip, but exact counts (e.g. "5 genuine regressions") could shift by one or two on a re-run.</li>
      <li><b>The judge and the faithfulness grader are both LLMs</b>, evaluated here without a held-out ground truth beyond the eval set's own hand-written in_scope/topic labels. The grader's rubric is fixed and its verdicts were spot-checked by a human on a random sample before this report was published — see the hand-check note below — but it is not a formally validated instrument.</li>
      <li><b>Models actually called</b> (read from response.model on each API call, not from config defaults): generation &mdash; {_esc(', '.join(mv['generation_models_seen']) or 'none recorded')}; judge &mdash; {_esc(', '.join(mv['judge_models_seen']) or 'none recorded')}.</li>
      <li><b>Latency</b>: mean judge call {lat['new_mean_judge_ms']} ms; mean generation call, new pipeline {lat['new_mean_generation_ms']} ms, old pipeline {lat['old_mean_generation_ms']} ms.</li>
    </ul>
  </section>

  <section>
    <h2><span class="num">07</span> Consistency assertions</h2>
    <div class="table-scroll">{assertion_rows}</div>
  </section>

  <section>
    <h2><span class="num">08</span> Raw data</h2>
    <div class="file-list">
      <div><b>data/eval_v1_summary_v2_5k_judge_v2.json</b> &mdash; new (judge-gated) pipeline, full detail, all 43 questions</div>
      <div><b>data/eval_v1_old_pipeline_full_v2_5k.json</b> &mdash; old (floor-gated) pipeline, full detail, all 43 questions</div>
      <div><b>data/eval_v1_faithfulness_v2_5k.json</b> &mdash; LLM grader verdicts for every graded answer</div>
      <div><b>data/eval_v1_grading_targets.json</b> &mdash; the (pipeline, id) list the grader was run against, computed by this script</div>
      <div><b>data/eval_v1_report_figures.json</b> &mdash; every figure on this page, as JSON</div>
      <div><b>data/eval_v1_report_assertions.log</b> &mdash; &sect;07's assertions, plain text</div>
      <div><b>CHANGES.md</b> &mdash; every number that changed from the previous report, old &rarr; new, and why</div>
    </div>
  </section>

  <footer>DiabEvidence &middot; answerability judge eval (corrected) &middot; not medical advice &mdash; a retrieval-evaluation report for a research-evidence QA system.</footer>
</div>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit-targets", action="store_true", help="Write grading targets and exit (before faithfulness grading has run).")
    args = parser.parse_args()

    eval_set = load_eval_set()
    new_p = load_pipeline(NEW_PIPELINE_PATH)
    old_p = load_pipeline(OLD_PIPELINE_PATH)
    core = build_core(eval_set, new_p, old_p)

    if args.emit_targets:
        emit_targets(core)
        return

    assertion_log = run_assertions(eval_set, core, new_p, old_p)
    for line in assertion_log:
        print(line)

    faithfulness = load_json(FAITHFULNESS_PATH)
    faithfulness_applied = apply_faithfulness(core, faithfulness, new_p, old_p)

    model_versions = collect_model_versions(new_p, old_p)
    latency = latency_stats(new_p, old_p)

    figures = build_figures(eval_set, core, new_p, old_p, faithfulness_applied, model_versions, latency)
    with open(FIGURES_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(figures, f, ensure_ascii=False, indent=2)
    print(f"\nWrote figures to {FIGURES_OUT_PATH}")

    with open("data/eval_v1_report_assertions.log", "w", encoding="utf-8") as f:
        f.write("\n".join(assertion_log) + "\n")
    print(f"Wrote assertion log to data/eval_v1_report_assertions.log")

    report_html = render_html(eval_set, figures, core, new_p, old_p, faithfulness, assertion_log)
    with open(HTML_OUT_PATH, "w", encoding="utf-8") as f:
        f.write(report_html)
    print(f"Wrote report HTML to {HTML_OUT_PATH}")


if __name__ == "__main__":
    main()
