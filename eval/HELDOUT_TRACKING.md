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
| Held-out run | running | `data/eval_heldout_v1_v2_5k.json` | |
| Faithfulness grading | not started | `eval/hand_grading_heldout_v1.csv` | |
| Report section appended | not started | `eval/REPORT_twostage_judge.md` | |

## Open items
- Mano's step-3 instructions were cut off at "If a run crashes, fix"; the rest is still to come.

## Deviations log
- 2026-09-28: **Questions drafted by Claude, not Mano**, at Mano's explicit request ("generate new 16 questions and run
  them") after the template stayed empty. This weakens the held-out claim: the drafter knew the failure modes the revised
  prompts target. Mitigations: written before any corpus lookup, checked only for near-duplicates of the dev set (max
  token overlap 0.14 after replacing one question), committed before the run, pipeline unchanged. Treat the results as
  semi-independent, not a clean held-out measurement.
