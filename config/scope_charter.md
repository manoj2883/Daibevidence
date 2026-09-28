<!--
SOURCE OF TRUTH for what DiabEvidence covers.

This file is the single, editable definition of scope for both judge stages
(src/rag/scope_judge.py Stage A, src/rag/evidence_judge.py Stage B) and the
frontend's "what this system covers" refusal text. It is loaded at startup
and read verbatim into the Stage A prompt — nothing in this file's content
is duplicated or paraphrased in code. Edit this file to change what counts
as in-charter, adjacent, or unrelated; no code change is needed for a
wording or example change, only for a change to which of the two judge
prompts consumes which section.
-->

# DiabEvidence scope charter

DiabEvidence answers questions from adults with type 2 diabetes, or people caring for them,
about four specific areas:

1. **Diet and nutrition** — what to eat, dietary patterns, and their effect on diabetes.
2. **Glycemic control** — blood sugar management, monitoring, and targets.
3. **Body composition and weight** — weight, body fat, and their relationship to diabetes.
4. **Diet and medication interactions** — how food, nutrients, or eating patterns interact
   with diabetes medications.

## In charter

A question is **in_charter** if it asks about one of the four areas above, for adults with
type 2 diabetes or people caring for them.

Examples:
- "Does a low-carbohydrate diet lower HbA1c in adults with type 2 diabetes?"
- "How much weight loss is typically needed to see improved insulin sensitivity?"
- "Can grapefruit juice affect how well metformin works?"

## Adjacent

A question is **adjacent** if it is clearly about diabetes, but outside the four areas above.
This includes (not an exhaustive list): type 1 or gestational diabetes specifics, exercise
programming, mental health and diabetes distress, complications (retinopathy, neuropathy,
nephropathy, foot care, cardiovascular disease), insulin dosing and pump/CGM hardware,
diabetic ketoacidosis, and insurance, disability, or licensing questions related to diabetes.

Examples:
- "What insulin-to-carbohydrate ratio should I use for a high-protein meal?" (insulin dosing)
- "How often should someone with diabetes get a dilated eye exam?" (complications/eye care)
- "What are the symptoms of diabetic ketoacidosis?" (DKA)

## Unrelated

A question is **unrelated** if it has no connection to diabetes at all.

Examples:
- "What is the capital of Australia?"
- "How do I refinance a car loan?"
- "What's a good recipe for banana bread?"

## Notes for maintainers

- This file is the only place scope is defined. If you change a category's description or
  examples here, both judge stages pick it up automatically at their next call — nothing else
  needs editing.
- The four in-charter areas above must stay in sync with `src/ingest/config.py`'s `TOPICS`
  dict, which drives what actually gets ingested into the corpus. If you add a fifth area
  here, ingestion needs to catch up before the corpus can actually answer it — the scope
  judge does not know what's in the corpus, only what this charter says the system is meant
  to cover (see `src/rag/scope_judge.py`'s docstring for why that separation matters).
