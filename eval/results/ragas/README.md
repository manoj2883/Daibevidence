# Ragas scores, dev set (2026-10-22)

43 dev questions (`eval/eval_set_v2.json`), latest pipeline outputs (Run B + sentence verifier). Held-out set not touched.
Judge `claude-haiku-5-5`, embeddings all-MiniLM-L6-v2 (local), ragas 0.4.3. Script: `scripts/run_ragas_eval.py`.

- `dataset.jsonl`: the 43 rows given to ragas (user_input, retrieved_contexts as full parent abstracts, response, reference).
- `scores.jsonl`: one line per (id, metric). 92 scores, 0 errors. 244 judge calls, 623k input / 126k output tokens.
- `report.md`, `summary.json`: the table, the lowest ids, agreement with the existing grades.
- `explain_34_13_25.txt`: the statements ragas faithfulness marked unsupported for three ids (`--explain`, a fresh judge call).

No Batch API: ragas sends one prompt per call and faithfulness chains two calls (statements, then verdicts), so the
Message Batches API does not fit without rewriting the metrics. Calls ran 4 at a time.

## Reading the numbers

**Faithfulness agreement with the existing grades is low (7/24 three-class, 8/24 binary) because the two measure
different things.** No answer scores 1.0 under ragas, so every existing SUPPORTED becomes PARTIALLY or UNSUPPORTED.
The `--explain` output shows what ragas counts against the answers:

1. Plain-language definitions added in patient mode ("HbA1c is a blood test showing average blood sugar over about three
   months", "a diabetic foot ulcer is an open sore"). The existing rubric explicitly does not penalize these.
2. The pipeline's own evidence-gap and population statements ("the studies do not address which footwear", "the review
   looked at mixed groups", "some studies focused on type 2"). Gap statements describe what the passages lack, so no
   passage states them; population labels come from metadata, not the abstract text.
3. The safety line ("talk to a doctor or pharmacist before changing any diabetes treatment").

Id 34 (0.14 vs SUPPORTED) is almost entirely 1 and 2: 1 of 7 statements is a finding, the other 6 are a definition, gap
or population statements. So ragas faithfulness here mostly measures how much of an answer is caveats and definitions, not
whether its findings are grounded. It is not a substitute for the existing grader without a rubric change (e.g.
excluding definition/gap/safety statements before the verdict step).

The existing grades are mixed too: 17 of 24 were by Claude Code by hand (credit exhausted at the time), 7 by the automated
Sonnet grader. Agreement is the same for both groups (29%).

**Context metrics.**
- `answered_partial` precision (0.771) is measured on the parents the evidence judge kept, which is a filtered set, so it is
  not comparable to a raw top-6 precision.
- `not_covered` rows score low on both (0.259 / 0.167). Those are the retrieved but unused parents, and low scores there
  are consistent with the pipeline declining correctly.
- Recall is low overall (0.348). The references cite specifics (named trials, effect sizes) that the retrieved abstracts
  often don't contain. Ids with recall 0.00 are worth a retrieval look.
- In-scope id 23 has no reference answer, so it has no context scores (n=23, not 24).
