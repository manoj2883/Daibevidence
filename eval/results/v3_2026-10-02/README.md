# v3 dev-set eval (2026-10-02)

43 dev questions (`eval/eval_set_v2.json`), patient mode. Held-out set not touched.

- Run A (`eval_v3_runA_v2_5k_noscope*`): v2_5k + background, no scope judge, dense top-6 chunks.
- Run B (`eval_v3_runB_full*`): full v3: query step, hybrid retrieval on v3_5k_recursive, full parents, ordering.
- `eval_v3_smoke.json`: 3 unscored smoke questions (Run B config).
- `*.INVALID_empty_answers.*`: outputs from before the contradiction-block parser fix; ids 14, 29, 32 (A)
  and 2, 17, 25 (B) had answered statuses with no answer text and were re-run after the fix.

Summary table and drug-name audit: `eval_v3_report.md`; analysis: `docs/DESIGN_DECISIONS.md` §9.
