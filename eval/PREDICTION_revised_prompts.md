# Prediction for the revised-prompt run (recorded before implementation)

Recorded 2026-09-28, before any code change for the revised Stage A / Stage B prompts, and before the run.

- **Run:** revised prompts (user-written, 2026-09-28) on the same 43 questions (`eval/eval_set_v2.json`, unmodified).
- **Status of these numbers:** development/regression only. Both prompts were written from failures on these
  43 questions, so results here are optimistic by construction and are not a held-out measurement.
- **Prediction (Mano):** ids 10, 13, 16, 18 become caveated answers, so in-scope answered goes from 10/24 to ~14/24 (58%).
- **Pre-run checks:**
  - 0b: neither Haiku judge enables extended thinking (scope max_tokens=300, evidence 1500); the previous
    run had 0 parse retries and 0 fail-closed events on every id.
  - 0c: the `v2_5k` chunk store (13,733 chunks) and the background tier (35) contain no "dawn phenomenon" text and no chunk
    mentioning a sulfonylurea together with alcohol/ethanol. Ids 9 and 20 are corpus gaps, not retrieval failures.
