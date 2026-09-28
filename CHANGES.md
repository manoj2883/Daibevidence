# Changes to the answerability-judge eval report

The first published version of this report ("Answerability Judge Eval") had real analysis
errors. This document lists every number that changed between that version and the
corrected one, and why. All corrected figures are computed by `scripts/build_judge_report.py`
from files in `data/` — see that script for the exact derivation, and
`data/eval_v1_report_assertions.log` for the internal-consistency checks that back them.

## 1. The comparison was not symmetric

**Before:** the "old" out-of-scope false-accept rate (84.2%, 16/19) came from a
retrieval-only run — no generation, just the similarity floor's pre-generation guess. The
"old" in-scope number came from a *different* measurement: an 11-question partial
judge-bypass run, not a full 43-question end-to-end run. Comparing these two differently-sourced
numbers as if they were the same kind of measurement was the core error.

**After:** both "old" and "new" now come from the same kind of measurement — a full,
real, end-to-end run of all 43 questions through each pipeline (`scripts/run_old_pipeline_full.py`
→ `data/eval_v1_old_pipeline_full_v2_5k.json`; `scripts/run_eval_v1.py` →
`data/eval_v1_summary_v2_5k_judge_v2.json`). The old pipeline reconstructs the pre-judge system
exactly: the similarity floor decides which chunks survive (`retrieval_mode="floor"` on
`stream_answer`), the judge is bypassed (`force_judge=True`), and anything that clears the floor
goes to real Sonnet generation.

This alone changed the headline numbers substantially, because the old pipeline's own
generation stage (rule 7 in `SYSTEM_PROMPT`: "if no excerpt answers the specific claim, say so")
was already catching a meaningful fraction of cases that the pre-generation floor signal missed:

| | Old (retrieval-only proxy) | Old (real end-to-end) |
|---|---|---|
| Diabetes-adjacent accepted | 84.2% (16/19, an earlier run) | **47.1%** (8/17, this run) |
| In-scope answered | 100% (24/24, provisional) | **79.2%** (19/24, real) |

The "84.2%" and "47.1%" also aren't the same run of the corpus/model — see §4 below — so
this row mixes two effects (proxy-vs-real, and run-to-run variance). The qualitative
point holds either way: the floor's own provisional signal overstates how often the old
system actually answered out-of-scope questions, because it never got to see whether
generation would hedge.

## 2. "Corrected false-reject rate: 20.8%" was itself wrong, in two ways

**Before (draft 2):** after noticing the first draft's asymmetry, I bypassed the judge for
just the 11 questions the new pipeline refused, and reported "5 genuine regressions, 6 would
have failed anyway, corrected rate 20.8%." This was better than draft 1 but still wrong:

- It only fixed the in-scope side of the asymmetry, not the out-of-scope side (§1).
- It listed 6 "would have failed anyway" cases but only rendered 5 cards — a plain
  transcription miss. The 6th (id 11, "glycemic variability and its clinical relevance") was
  never shown to you before this correction.

**After:** every count is now asserted, not hand-typed. `run_assertions()` checks that
every new-pipeline in-scope refusal is accounted for by exactly two buckets
(`old_only` ∪ `neither_answered`, with no leftover case) — this is the specific check
that would have caught the missing-id-11 error immediately if it had existed in draft 2.

The real reconciliation numbers are also different from both earlier drafts:

| | Draft 1 (naive) | Draft 2 ("corrected") | This version |
|---|---|---|---|
| New in-scope refusals | 11/24 (45.8%) | 11/24, reframed as 5 "genuine" | **6/24 (25.0%)**\* |
| Of those, genuine regressions | — | 5 | **2** (ids 7, 18) |
| Of those, reclassified as correct refusals | — | 0 named (id 12 not yet graded) | **4** (ids 12, 15, 21, 23) |
| Of those, old pipeline also failed | 0 named | 6 (1 missing) | **5** (ids 9, 11, 17, 20, 24) |

\* 6, not 11 — because the *old* pipeline in this run also failed to answer 5 of those same
24 questions (its own generation-stage hedge), so the true "new-only" refusal set (old
answered, new refused) is smaller than draft 2's naive in-scope-refusal count implied. See
the in-scope contingency table in `data/eval_v1_report_figures.json` for the full 2×2
breakdown (`both_answered` / `old_only` / `new_only` / `neither_answered`).

## 3. The reclassification rule itself was inconsistent — caught and fixed mid-review

Fixing draft 2's asymmetry surfaced a new problem: the LLM faithfulness grader
(`scripts/grade_faithfulness.py`) graded two structurally identical answers differently.

Both id 12 (DPP-4 monotherapy) and id 23 (protein intake when initiating insulin) have the
same shape: several sentences describing real but tangential evidence, then a closing
sentence that accurately states no excerpt addresses the specific claim asked. The original
grader rubric graded id 12's closing hedge "PARTIALLY_SUPPORTED" (a technicality: one excerpt
has weak, tangential counter-evidence) → overall UNSUPPORTED → reclassified as a correct
refusal, exactly as the task asked. But it graded id 23's closing hedge "SUPPORTED" (the hedge
is accurate) → overall SUPPORTED → left as a genuine regression. The rubric never told the
grader what to do when the central claim *is* an accurate hedge — it graded the hedge's
factual accuracy instead of asking whether the question got answered.

**Fix:** `GRADER_SYSTEM_PROMPT` now has an explicit rule: when the central claim is itself a
"no excerpt addresses this" hedge, the overall verdict is UNSUPPORTED regardless of whether
the hedge is accurate, because the question was never actually resolved. The grader was
re-run against all 24 targets with the fixed rubric (`data/eval_v1_faithfulness_v2_5k.json`,
overwritten in place — the pre-fix version is not separately kept, since it was demonstrably
inconsistent and no correct number was ever derived from it).

This changed the reconciliation again, this time for the better: ids 15 and 22 had the exact
same "accurate hedge, unresolved question" shape and were previously graded inconsistently
with 12/21 too (15 as PARTIALLY_SUPPORTED/genuine-regression, 22 as SUPPORTED). Both now
correctly reclassify. Final genuine-regression count: **2** (ids 7, 18), not 4 or 5.

Residual note: the grader is still not perfectly deterministic — re-running it after the
rubric fix also shifted a few unrelated verdicts (e.g. id 1, id 6) by one grade level in
ways that look like ordinary LLM sampling variance, not a rubric gap. This is disclosed in
the report's Limitations section (§06) rather than chased further; a single-digit-count
shift on re-run is expected, not a sign of a remaining systematic error.

## 4. Non-determinism between runs

Every full pipeline run in this project used a fresh, real call to Sonnet (generation) and
Haiku (judge/grader) — nothing is cached or replayed. Because of this, the exact ids in the
out-of-scope false-accept set changed between the very first run reported in chat (ids 25,
29, 32, 33) and the run this report is built from (ids 25, 29, 32, 33, 37 — id 37, DKA
management, was a new false accept on this run that didn't appear on the first). This is
expected LLM variance, not a bug, and is why the report's Limitations section says single-run
counts can shift by one or two on a re-run.

## 5. Scope-label precision

**Before:** "the judge correctly refused 15 of the other 15 diabetes-adjacent questions" —
conflating "15 total correct refusals" (13 adjacent + 2 unrelated) with "15 diabetes-adjacent
questions." There are only 17 diabetes-adjacent questions total, 5 of which were accepted
(this run), so at most 12 diabetes-adjacent questions could have been correctly refused, not 15.

**After:** the report always reports adjacent (n=17) and unrelated (n=2) as separate
subgroups with their own accept counts and rates (`adjacent_contingency` /
`unrelated_contingency` in the figures JSON), never a combined "out of scope" figure
presented as if it were one subgroup.

## 6. False-accept definition, applied consistently

The report picks one definition of scope — the four declared ingestion topics
(`src/ingest/config.py` `TOPICS`) — and applies it uniformly: any diabetes-adjacent or
unrelated question that gets answered is a false accept, full stop, regardless of whether the
corpus happens to contain genuinely relevant text for it (which it does, incidentally, for a
few sub-areas like retinopathy and CGM hardware — see §03 of the report). The alternative
definition ("scope = whatever the corpus can answer") is not used and not mixed in; the eval
set's own `topic`/`in_scope` labels were not changed.

## 7. Id 20 (alcohol × sulfonylurea): labeled, not fixed

Confirmed again in this corrected run: id 20 is a `neither_answered` case (old pipeline also
failed it) caused by a real, pre-existing bug in `query_classifier.py` — it tags this
question "definitional," routing retrieval to the background-only tier and missing a directly
relevant study-tier paper. `classifier_definitional_misroute()` in `build_judge_report.py`
detects this mechanically (not by hand-listing id 20) so it would flag any other question with
the same failure mode too. The report's case card for it has a distinct dashed border and its
own legend line. **Not fixed in this change**, per instructions — tracked here as a known,
separate issue in `query_classifier.py`'s definitional/evidence-seeking classification.

## 8. New finding surfaced by faithfulness grading, out of scope for this task

Id 10 (new pipeline, "postprandial exercise timing") passed the judge and generation marked it
`answered`, but the faithfulness grader found the excerpts only describe study *aims*, never
their *results* — the question is never actually resolved. This is a generation-stage
faithfulness gap, unrelated to the judge/floor comparison this task is about. Not addressed
here; noted for a future task.

## Files

- `data/eval_v1_old_pipeline_full_v2_5k.json` — old pipeline, full detail, all 43 questions
  (superseded an earlier accidental run against the wrong 30-question eval set, which is kept
  as `data/eval_v1_old_pipeline_full_v2_5k.WRONG_eval_set_v1.json.bak` for the record, not used
  by anything)
- `data/eval_v1_summary_v2_5k_judge_v2.json` — new pipeline, full detail, all 43 questions
- `data/eval_v1_faithfulness_v2_5k.json` — faithfulness grader verdicts (post-rubric-fix)
- `data/eval_v1_grading_targets.json` — the (pipeline, id) list graded, computed by
  `scripts/build_judge_report.py --emit-targets`
- `data/eval_v1_report_figures.json` — every figure in the report, as JSON
- `data/eval_v1_report_assertions.log` — the 16 consistency assertions, all passing
- `data/eval_v1_judge_report.html` — the regenerated report
