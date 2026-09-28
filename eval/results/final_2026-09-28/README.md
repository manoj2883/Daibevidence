# DiabEvidence judge evaluation: final results (2026-09-28)

Copies of the run outputs (the originals live in the gitignored `data/`). Full analysis: `eval/REPORT_twostage_judge.md`.

| Run | Questions | In-scope answered | Files |
|---|---|---|---|
| Two-stage judge, original prompts | 43 dev | 10/24 (41.7%) | `eval_v2_judge_twostage_v2_5k*.json` |
| Revised prompts, dev/regression | 43 dev | 14/24 (58.3%), matching the recorded prediction | `eval_v3_revised_prompts_v2_5k*.json`, `eval_v3_run.log` |
| Revised prompts, held-out | 16 new | 3/8 (37.5%, 95% CI 13.7–69.4%) | `eval_heldout_v1_v2_5k*.json`, `eval_heldout_v1_run*.log` |

Held-out, other groups: adjacent 0/5 answered, unrelated 0/3; scope correct on 15/16 (id 107 missed).
Held-out faithfulness of answered cases: 105 PARTIALLY_SUPPORTED, 106 SUPPORTED, 108 UNSUPPORTED.

Caveats:
- The dev numbers are optimistic: both revised prompts were written from failures on those 43 questions.
- The held-out questions were drafted by Claude at Mano's request, so they're semi-independent (see `eval/HELDOUT_TRACKING.md`).
- The held-out run was interrupted by an API usage limit and resumed. `eval_heldout_v1_v2_5k.INVALID_api_usage_limit.json` is the
  invalid first pass (ids 109–116 failed closed), kept for the audit trail only.
- With n=8/5/3 per group, every interval is wide.
