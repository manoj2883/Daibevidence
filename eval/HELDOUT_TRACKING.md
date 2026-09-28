# Held-out evaluation: tracking log

Single place to track the held-out run of the revised judge prompts. Update the status table as each step completes.

## Protocol (fixed before any held-out question exists)
- Questions and labels are written by Mano, not by Claude, and committed unmodified to `eval/heldout_v1.json`.
  Template: `eval/heldout_v1_TEMPLATE.json` (16 slots, ids 101–116 so they never collide with the dev set's 1–43).
- Pipeline frozen at commit `b82d8c2` (revised prompts `978771a` / `7fcce3d`, charter as committed). No prompt, charter or
  code change between receiving the questions and finishing the run.
- Run exactly once: `python -u -m scripts.run_eval_v2_twostage --namespace v2_5k --questions eval/heldout_v1.json
  --out data/eval_heldout_v1_v2_5k.json --label "held-out v1, revised prompts"`. The runner is resumable, so a crash is
  resumed rather than re-run. A code fix after a crash is recorded below with its reason.
- Grade: `python -m scripts.grade_faithfulness data/eval_heldout_v1_v2_5k.json --csv-out eval/hand_grading_heldout_v1.csv`
- Report: `python -m scripts.report_revised_run --mode heldout --results data/eval_heldout_v1_v2_5k.json
  --questions eval/heldout_v1.json` appends to `eval/REPORT_twostage_judge.md`.
- These are the headline numbers. The 43-question results (section 10 of the report) are development/regression only.

## Status
| Step | Status | Commit / file | Notes |
|---|---|---|---|
| Dev/regression run on the 43 | done | `f74c6b7` | 14/24 in-scope answered vs predicted 14/24 |
| Held-out questions | done (Claude-drafted, see deviations) | `eval/heldout_v1.json` | 8 in_scope / 5 adjacent / 3 unrelated |
| Held-out run | **partial: 8/16 valid, blocked** | `data/eval_heldout_v1_v2_5k.json` | |
| Faithfulness grading | **blocked** (3 grades invalid) | `eval/hand_grading_heldout_v1.csv` | |
| Report section appended | not started | `eval/REPORT_twostage_judge.md` | |

## Open items
- **API usage limit reached mid-run (2026-09-28 ~13:15). Access returns 2026-10-01 00:00 UTC.** Ids 101-108 completed
  every stage and are kept in `data/eval_heldout_v1_v2_5k.json.partial.jsonl`. Ids 109-116 failed closed at Stage A on
  the API error ("Scope judge API call failed") and are invalid; all 3 faithfulness grades (105, 106, 108) are invalid
  for the same reason. Invalid outputs are kept as `*.INVALID_api_usage_limit.json`.
  To finish, after the reset, re-run the same run command (it resumes at 109), then the grading and report commands.
  Or raise the usage limit in the Anthropic Console to finish sooner.
- Mano's step-3 instructions were cut off at "If a run crashes, fix"; the rest is still to come.

## Deviations log
- 2026-09-28: **Questions drafted by Claude, not Mano**, at Mano's explicit request ("generate new 16 questions and run
  them") after the template stayed empty. This weakens the held-out claim: the drafter knew the failure modes the revised
  prompts target. Mitigations: written before any corpus lookup, checked only for near-duplicates of the dev set (max
  token overlap 0.14 after replacing one question), committed before the run, pipeline unchanged. Treat the results as
  semi-independent, not a clean held-out measurement.
- 2026-09-28: Run interrupted by the API usage limit (see Open items). Runner fixed so a judge API failure counts as an
  error, is kept out of the checkpoint, and blocks writing the final file; previously such failures were silently
  recorded as `out_of_scope` with "Errors: 0". No pipeline (prompt, charter, judge) code changed.
