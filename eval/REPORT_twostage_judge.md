# Two-stage answerability judge — eval report

Single run, `eval/eval_set_v2.json` (43 questions, unmodified), production pipeline (`src.rag.chain.stream_answer`, no diagnostic overrides). Every number below is computed by `scripts/build_report_v2.py` from files in `data/` and `eval/` — see that script for the exact derivation, and §7 for the consistency assertions it ran before writing anything.

## 1. What changed and why

The single boolean answerability judge (`src/rag/judge.py`) conflated two decisions — is this question in the corpus's subject matter (scope), and do the retrieved chunks contain enough to answer it (evidence) — into one yes/no, and accepted on topical overlap (ids 1, 10, 22 were judged answerable at high similarity and produced UNSUPPORTED answers). It's replaced by two independent stages: Stage A (`src/rag/scope_judge.py`) decides scope from the question alone, before any retrieval; Stage B (`src/rag/evidence_judge.py`) audits retrieved passages per-proposition (DIRECT/PARTIAL/ABSENT) rather than a single answerable boolean. Five statuses replace the old four-state enum: `answered`, `answered_partial` (generation restricted to DIRECT passages, names the gap), `in_scope_no_evidence` (a valid in-charter question the corpus has no evidence for — never called out of scope), `answered_adjacent` (evidence sufficient, question outside the four core areas), `out_of_scope` (unrelated, or adjacent + insufficient/partial). Full detail: `RECON.md`.

## 2. Answer/refusal rates by label, with 95% Wilson intervals

| Label | n | New: answered | New: in_scope_no_evidence | New: out_of_scope | Old floor: answered | Prev boolean judge: answered |
|---|---|---|---|---|---|---|
| in_scope | 24 | 41.7% (95% CI 24.5–61.2%) | 29.2% (95% CI 14.9–49.2%) | 29.2% (95% CI 14.9–49.2%) | 79.2% (95% CI 59.5–90.8%) | 54.2% (95% CI 35.1–72.1%) |
| adjacent | 17 | 0.0% (95% CI 0.0–18.4%) | 0.0% (95% CI 0.0–18.4%) | 100.0% (95% CI 81.6–100.0%) | 47.1% (95% CI 26.2–69.0%) | 29.4% (95% CI 13.3–53.1%) |
| unrelated | 2 | 0.0% (95% CI 0.0–65.8%) | 0.0% (95% CI 0.0–65.8%) | 100.0% (95% CI 34.2–100.0%) | 0.0% (95% CI 0.0–65.8%) | 0.0% (95% CI 0.0–65.8%) |

Interval widths at n=17 and especially n=2 are wide — see §8 Limitations before reading any single-run difference above as a real effect.

## 3. Confusion matrix: Stage A scope vs. eval-set label

| Stage A scope \ eval label | in_scope | adjacent | unrelated |
|---|---|---|---|
| in_charter | [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 15, 18, 19, 21, 22, 24] (n=17) | [] (n=0) | [] (n=0) |
| adjacent | [10, 12, 13, 16, 17, 20, 23] (n=7) | [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41] (n=17) | [] (n=0) |
| unrelated | [] (n=0) | [] (n=0) | [42, 43] (n=2) |

## 4. Faithfulness distribution of answered cases, by label

| Label | SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED |
|---|---|---|---|
| in_scope | 6 | 3 | 1 |
| adjacent | 0 | 0 | 0 |
| unrelated | 0 | 0 | 0 |

## 5. Per-question table (all 43)

| id | label | topic | top-1 score | old status | prev judge status | new scope | new evidence verdict | new status | faithfulness |
|---|---|---|---|---|---|---|---|---|---|
| 1 | in_scope | diet_nutrition | 0.852 | answered | answered | in_charter | sufficient | answered | SUPPORTED |
| 2 | in_scope | diet_nutrition | 0.769 | answered | answered | in_charter | partial | answered_partial | SUPPORTED |
| 3 | in_scope | diet_nutrition | 0.839 | answered | answered | in_charter | sufficient | answered | SUPPORTED |
| 4 | in_scope | diet_nutrition | 0.828 | answered | answered | in_charter | sufficient | answered | SUPPORTED |
| 5 | in_scope | diet_nutrition | 0.762 | answered | answered | in_charter | sufficient | answered | SUPPORTED |
| 6 | in_scope | diet_nutrition | 0.791 | answered | answered | in_charter | partial | answered_partial | PARTIALLY_SUPPORTED |
| 7 | in_scope | glycemic_control | 0.734 | answered | out_of_scope | in_charter | partial | answered_partial | UNSUPPORTED |
| 8 | in_scope | glycemic_control | 0.862 | answered | answered | in_charter | partial | answered_partial | PARTIALLY_SUPPORTED |
| 9 | in_scope | glycemic_control | 0.120 | no_evidence_for_claim | out_of_scope | in_charter | insufficient | in_scope_no_evidence | n/a |
| 10 | in_scope | glycemic_control | 0.759 | answered | answered | adjacent | partial | out_of_scope | n/a |
| 11 | in_scope | glycemic_control | 0.469 | no_evidence_for_claim | out_of_scope | in_charter | insufficient | in_scope_no_evidence | n/a |
| 12 | in_scope | glycemic_control | 0.812 | answered | out_of_scope | adjacent | insufficient | out_of_scope | n/a |
| 13 | in_scope | body_composition_weight | 0.798 | answered | answered | adjacent | partial | out_of_scope | n/a |
| 14 | in_scope | body_composition_weight | 0.844 | answered | answered | in_charter | sufficient | answered | SUPPORTED |
| 15 | in_scope | body_composition_weight | 0.662 | answered | out_of_scope | in_charter | insufficient | in_scope_no_evidence | n/a |
| 16 | in_scope | body_composition_weight | 0.824 | answered | answered | adjacent | partial | out_of_scope | n/a |
| 17 | in_scope | body_composition_weight | 0.351 | no_evidence_for_claim | out_of_scope | adjacent | insufficient | out_of_scope | n/a |
| 18 | in_scope | body_composition_weight | 0.692 | answered | out_of_scope | in_charter | insufficient | in_scope_no_evidence | n/a |
| 19 | in_scope | diet_medication_interaction | 0.737 | answered | answered | in_charter | partial | answered_partial | PARTIALLY_SUPPORTED |
| 20 | in_scope | diet_medication_interaction | 0.526 | out_of_scope | out_of_scope | adjacent | insufficient | out_of_scope | n/a |
| 21 | in_scope | diet_medication_interaction | 0.642 | answered | out_of_scope | in_charter | insufficient | in_scope_no_evidence | n/a |
| 22 | in_scope | diet_medication_interaction | 0.720 | answered | answered | in_charter | insufficient | in_scope_no_evidence | n/a |
| 23 | in_scope | diet_medication_interaction | 0.680 | answered | out_of_scope | adjacent | insufficient | out_of_scope | n/a |
| 24 | in_scope | diet_medication_interaction | 0.618 | no_evidence_for_claim | out_of_scope | in_charter | insufficient | in_scope_no_evidence | n/a |
| 25 | adjacent | out_of_scope_diabetes_adjacent | 0.716 | answered | answered | adjacent | partial | out_of_scope | n/a |
| 26 | adjacent | out_of_scope_diabetes_adjacent | 0.628 | no_evidence_for_claim | out_of_scope | adjacent | insufficient | out_of_scope | n/a |
| 27 | adjacent | out_of_scope_diabetes_adjacent | 0.543 | answered | out_of_scope | adjacent | insufficient | out_of_scope | n/a |
| 28 | adjacent | out_of_scope_diabetes_adjacent | 0.755 | answered | out_of_scope | adjacent | insufficient | out_of_scope | n/a |
| 29 | adjacent | out_of_scope_diabetes_adjacent | 0.673 | answered | answered | adjacent | partial | out_of_scope | n/a |
| 30 | adjacent | out_of_scope_diabetes_adjacent | 0.275 | out_of_scope | out_of_scope | adjacent | insufficient | out_of_scope | n/a |
| 31 | adjacent | out_of_scope_diabetes_adjacent | 0.456 | out_of_scope | out_of_scope | adjacent | insufficient | out_of_scope | n/a |
| 32 | adjacent | out_of_scope_diabetes_adjacent | 0.691 | answered | answered | adjacent | partial | out_of_scope | n/a |
| 33 | adjacent | out_of_scope_diabetes_adjacent | 0.830 | answered | answered | adjacent | partial | out_of_scope | n/a |
| 34 | adjacent | out_of_scope_diabetes_adjacent | 0.717 | no_evidence_for_claim | out_of_scope | adjacent | insufficient | out_of_scope | n/a |
| 35 | adjacent | out_of_scope_diabetes_adjacent | 0.570 | no_evidence_for_claim | out_of_scope | adjacent | insufficient | out_of_scope | n/a |
| 36 | adjacent | out_of_scope_diabetes_adjacent | 0.615 | no_evidence_for_claim | out_of_scope | adjacent | insufficient | out_of_scope | n/a |
| 37 | adjacent | out_of_scope_diabetes_adjacent | 0.707 | answered | answered | adjacent | partial | out_of_scope | n/a |
| 38 | adjacent | out_of_scope_diabetes_adjacent | 0.736 | answered | out_of_scope | adjacent | insufficient | out_of_scope | n/a |
| 39 | adjacent | out_of_scope_diabetes_adjacent | 0.599 | no_evidence_for_claim | out_of_scope | adjacent | insufficient | out_of_scope | n/a |
| 40 | adjacent | out_of_scope_diabetes_adjacent | 0.569 | no_evidence_for_claim | out_of_scope | adjacent | insufficient | out_of_scope | n/a |
| 41 | adjacent | out_of_scope_diabetes_adjacent | 0.585 | no_evidence_for_claim | out_of_scope | adjacent | insufficient | out_of_scope | n/a |
| 42 | unrelated | out_of_scope_unrelated | 0.000 | out_of_scope | out_of_scope | unrelated | None | out_of_scope | n/a |
| 43 | unrelated | out_of_scope_unrelated | 0.000 | out_of_scope | out_of_scope | unrelated | None | out_of_scope | n/a |

## 6. Specifically requested: ids 1, 7, 10, 18, 20, 22, 25, 33, 37

| id | label | topic | top-1 score | old status | prev judge status | new scope | new evidence verdict | new status | faithfulness |
|---|---|---|---|---|---|---|---|---|---|
| 1 | in_scope | diet_nutrition | 0.852 | answered | answered | in_charter | sufficient | answered | SUPPORTED |
| 7 | in_scope | glycemic_control | 0.734 | answered | out_of_scope | in_charter | partial | answered_partial | UNSUPPORTED |
| 10 | in_scope | glycemic_control | 0.759 | answered | answered | adjacent | partial | out_of_scope | n/a |
| 18 | in_scope | body_composition_weight | 0.692 | answered | out_of_scope | in_charter | insufficient | in_scope_no_evidence | n/a |
| 20 | in_scope | diet_medication_interaction | 0.526 | out_of_scope | out_of_scope | adjacent | insufficient | out_of_scope | n/a |
| 22 | in_scope | diet_medication_interaction | 0.720 | answered | answered | in_charter | insufficient | in_scope_no_evidence | n/a |
| 25 | adjacent | out_of_scope_diabetes_adjacent | 0.716 | answered | answered | adjacent | partial | out_of_scope | n/a |
| 33 | adjacent | out_of_scope_diabetes_adjacent | 0.830 | answered | answered | adjacent | partial | out_of_scope | n/a |
| 37 | adjacent | out_of_scope_diabetes_adjacent | 0.707 | answered | answered | adjacent | partial | out_of_scope | n/a |

Id 20 (the classifier bug): fixed in `src/rag/query_classifier.py` before this run — see `RECON.md` §7 for the root cause and `git log` for the fix commit. Its row above reflects the fixed classifier.

## 7. Latency and token cost per stage, vs. the previous (boolean-judge) pipeline

- Stage A (scope): mean 1111.9ms, p95 1294.4ms
- Retrieval: mean 146.0ms, p95 194.5ms
- Stage B (evidence): mean 3630.1ms, p95 4658.3ms
- Generation (this run): mean 26020.9ms, p95 33337.8ms
- Generation (previous, floor-gated pipeline): mean 36571.5ms, p95 57577.3ms
- Total (this run): mean 31384.8ms, p95 38768.9ms
- Mean generation tokens, this run: 4197.2 in / 2714.5 out
- Mean generation tokens, previous pipeline: 5695.6 in / 2863.2 out

Models actually called (read from each response's own `.model` field, not config defaults): scope — claude-haiku-4-5-20251001; evidence — claude-haiku-4-5-20251001; generation — claude-sonnet-5. Faithfulness grader model: see `data/eval_v2_judge_twostage_v2_5k_faithfulness.json`'s `grader_model` field.

## 8. Limitations

- **n=43 is small, and n=2 for the unrelated group is not enough to estimate a rate at all** — read the Wilson intervals in §2 literally: most of them overlap across old/prev/new, meaning most apparent differences between pipelines in this report are not statistically distinguishable from noise at this sample size. Treat directional agreement across several ids as more informative than any single rate.
- **This eval set was used to diagnose the exact problems this rebuild fixes** (ids 1, 7, 10, 18, 20, 22, 25, 33, 37 were named in the task specification itself, based on the previous pipeline's known failures on them). Numbers in this report are optimistic for that reason — they measure whether the fix works on the cases known to motivate it, not on held-out questions the rebuild was never informed by. Confirming this on a genuinely new question set is the natural next step, not done here.
- **Faithfulness-grader independence**: the grader uses a different model (Sonnet, via `src.rag.chain.get_model()`) and independently-worded rubric from Stage A/B (both Haiku, via `src.rag.judge.get_judge_model()`) — verified by a unit test asserting zero shared distinctive vocabulary between the two prompts. This makes agreement between the grader and Stage B more informative than if they shared a model, but it does not make the grader a validated ground truth instrument — it is itself an LLM judgment, unreplicated by a second grading pass or a human rater beyond the `eval/hand_grading.csv` spot-check.
- **No prompt iteration, and the run history in full**: neither judge prompt was changed at any point after implementation. The pipeline eval completed exactly once, but it was *launched* four times: attempt 1 was aborted when Stage B's model wrapped its JSON in a code fence followed by prose, which the parser could not recover — fixed in parsing code only (`extract_json_object`, commit e1444aa), no prompt text touched; attempt 2 was killed by the OS for low memory; attempt 3 was killed by a machine restart. None of the aborted attempts wrote a results file, so no results were seen before the completed run. Separately, two bugs were fixed before any attempt (the `direct`/`partial` chunk-ref parsing accepting both int and "C{n}" string forms, and the id-20 classifier misroute), both caught by direct testing.
- **The faithfulness grader was run twice.** The first pass returned 6 of 10 grades as parse failures (fail-closed to UNSUPPORTED): the grading model emits a thinking block that counts against `max_tokens`, and 1500 was not enough, truncating or emptying the JSON. That pass is preserved as `data/eval_v2_judge_twostage_v2_5k_faithfulness.INVALID_max_tokens.json`. The fix raised `max_tokens` to 8000 and made a max_tokens stop an explicit grading error; the grader prompt was not changed, and the pipeline outputs being graded were not re-run. The grades in §4 are from the second pass, which had zero grading errors.
- **The grader is not deterministic.** During diagnosis of the failure above, a one-off grading call on id 1 returned PARTIALLY_SUPPORTED; the official second pass graded it SUPPORTED. Single grades near the SUPPORTED/PARTIALLY_SUPPORTED boundary should be read as uncertain — this is what `eval/hand_grading.csv` is for.
- **Result flagged, not tuned: Stage A labels 7 of 24 in-scope questions `adjacent`** (ids 10, 12, 13, 16, 17, 20, 23 — see §3), which routes them to `out_of_scope` and accounts for all 7 in-scope refusals in §2. This includes id 20: the classifier fix is in place, but Stage A now refuses it independently. Per the task's rule this was reported, not fixed by editing the scope prompt or charter against these 43 questions. Whether the charter wording or the Stage A prompt is the cause needs a decision from the maintainer and a held-out set to test the change on.
- **Single run, no variance estimate** beyond the Wilson intervals on the rates themselves — every judge/evidence/generation/grading call is a real, non-deterministic LLM call (this SDK build has no `temperature` parameter to reduce variance with — see `src/rag/scope_judge.py`'s note). Individual ids could grade differently on a re-run.

## 9. Consistency assertions

```
[PASS] 43 questions total
[PASS] 24 in_scope
[PASS] 17 adjacent
[PASS] 2 unrelated
[PASS] labels partition all 43 with no overlap
[PASS] new run covers all 43 ids
[PASS] id 1 has a valid status or a recorded error
[PASS] id 2 has a valid status or a recorded error
[PASS] id 3 has a valid status or a recorded error
[PASS] id 4 has a valid status or a recorded error
[PASS] id 5 has a valid status or a recorded error
[PASS] id 6 has a valid status or a recorded error
[PASS] id 7 has a valid status or a recorded error
[PASS] id 8 has a valid status or a recorded error
[PASS] id 9 has a valid status or a recorded error
[PASS] id 10 has a valid status or a recorded error
[PASS] id 11 has a valid status or a recorded error
[PASS] id 12 has a valid status or a recorded error
[PASS] id 13 has a valid status or a recorded error
[PASS] id 14 has a valid status or a recorded error
[PASS] id 15 has a valid status or a recorded error
[PASS] id 16 has a valid status or a recorded error
[PASS] id 17 has a valid status or a recorded error
[PASS] id 18 has a valid status or a recorded error
[PASS] id 19 has a valid status or a recorded error
[PASS] id 20 has a valid status or a recorded error
[PASS] id 21 has a valid status or a recorded error
[PASS] id 22 has a valid status or a recorded error
[PASS] id 23 has a valid status or a recorded error
[PASS] id 24 has a valid status or a recorded error
[PASS] id 25 has a valid status or a recorded error
[PASS] id 26 has a valid status or a recorded error
[PASS] id 27 has a valid status or a recorded error
[PASS] id 28 has a valid status or a recorded error
[PASS] id 29 has a valid status or a recorded error
[PASS] id 30 has a valid status or a recorded error
[PASS] id 31 has a valid status or a recorded error
[PASS] id 32 has a valid status or a recorded error
[PASS] id 33 has a valid status or a recorded error
[PASS] id 34 has a valid status or a recorded error
[PASS] id 35 has a valid status or a recorded error
[PASS] id 36 has a valid status or a recorded error
[PASS] id 37 has a valid status or a recorded error
[PASS] id 38 has a valid status or a recorded error
[PASS] id 39 has a valid status or a recorded error
[PASS] id 40 has a valid status or a recorded error
[PASS] id 41 has a valid status or a recorded error
[PASS] id 42 has a valid status or a recorded error
[PASS] id 43 has a valid status or a recorded error
[PASS] id 9 in_scope_no_evidence message never says 'out of scope'
[PASS] id 11 in_scope_no_evidence message never says 'out of scope'
[PASS] id 15 in_scope_no_evidence message never says 'out of scope'
[PASS] id 18 in_scope_no_evidence message never says 'out of scope'
[PASS] id 21 in_scope_no_evidence message never says 'out of scope'
[PASS] id 22 in_scope_no_evidence message never says 'out of scope'
[PASS] id 24 in_scope_no_evidence message never says 'out of scope'
[PASS] every graded id was actually answered
[PASS] every answered id was graded (grader ran on every answered case, not a subset)
```

## 10. Revised prompts: development/regression run on the 43 (2026-09-28)

Development/regression numbers only: both revised prompts were written from failures on these 43 questions.

| id | label | scope | evidence verdict | status | faithfulness |
|---|---|---|---|---|---|
| 1 | in_scope | in_charter | sufficient | answered | PARTIALLY_SUPPORTED |
| 2 | in_scope | in_charter | sufficient | answered | SUPPORTED |
| 3 | in_scope | in_charter | partial | answered_partial | SUPPORTED |
| 4 | in_scope | in_charter | sufficient | answered | SUPPORTED |
| 5 | in_scope | in_charter | sufficient | answered | SUPPORTED |
| 6 | in_scope | in_charter | partial | answered_partial | PARTIALLY_SUPPORTED |
| 7 | in_scope | in_charter | partial | answered_partial | UNSUPPORTED |
| 8 | in_scope | in_charter | partial | answered_partial | PARTIALLY_SUPPORTED |
| 9 | in_scope | in_charter | insufficient | in_scope_no_evidence | n/a |
| 10 | in_scope | in_charter | partial | answered_partial | UNSUPPORTED |
| 11 | in_scope | in_charter | partial | answered_partial | PARTIALLY_SUPPORTED |
| 12 | in_scope | in_charter | insufficient | in_scope_no_evidence | n/a |
| 13 | in_scope | in_charter | sufficient | answered | SUPPORTED |
| 14 | in_scope | in_charter | partial | answered_partial | SUPPORTED |
| 15 | in_scope | in_charter | insufficient | in_scope_no_evidence | n/a |
| 16 | in_scope | in_charter | partial | answered_partial | PARTIALLY_SUPPORTED |
| 17 | in_scope | in_charter | insufficient | in_scope_no_evidence | n/a |
| 18 | in_scope | in_charter | insufficient | in_scope_no_evidence | n/a |
| 19 | in_scope | adjacent | partial | out_of_scope | n/a |
| 20 | in_scope | in_charter | insufficient | in_scope_no_evidence | n/a |
| 21 | in_scope | in_charter | insufficient | in_scope_no_evidence | n/a |
| 22 | in_scope | in_charter | insufficient | in_scope_no_evidence | n/a |
| 23 | in_scope | in_charter | partial | answered_partial | UNSUPPORTED |
| 24 | in_scope | in_charter | insufficient | in_scope_no_evidence | n/a |

In-scope answered: 14/24 vs predicted 14/24 (ceiling 14; 10 are confirmed corpus gaps).

- Corpus-gap caveat: only ids [9, 20] were confirmed by searching the chunk store; the other 8 are gaps per Stage B's verdict only.
- Below ceiling: id 18 (in_scope_no_evidence): The core proposition requires a direct comparison of WC versus BMI as predictors of metabolic risk in type 2 diabetes, but C1 addresses prediabetes not diabetes, C2 discusses WWI not WC, C3 and C4 address outcomes other than metabolic risk prediction comparison, and C5 compares WHtR to BMI rather than WC to BMI.
- Below ceiling: id 19 (out_of_scope): The outcome concerns vitamin B12 status, which is a nutritional biomarker unrelated to glycemic control, body composition, diet, or diet-medication interactions on glucose metabolism.
- Answered beyond the prediction: ids [11, 23].
