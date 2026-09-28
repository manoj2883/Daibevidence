<!--
SOURCE OF TRUTH for what DiabEvidence covers.

This file is the single, editable definition of scope. src/rag/scope_judge.py (Stage A) reads
two sections of it into its prompt at startup:
  - "## Areas": the bulleted area list, inserted verbatim as the prompt's MAP step
  - "## Adjacent examples": the one-line example list, inserted into the prompt's ADJACENT step
Nothing in those two sections is duplicated in code. Edit them here to change what counts as
in_charter or adjacent; restart the process to pick up the edit. The four area keys
(glycemic_control, body_composition_weight, diet_nutrition, diet_medication_interaction) must
stay as written: the code validates the judge's "area" field against them, and startup fails
loudly if the Areas section is missing any of them.

The other sections (the rule, the population note, example questions) document intent for
maintainers and are not sent to the model.
-->

# DiabEvidence scope charter

DiabEvidence answers questions from adults with type 2 diabetes, or people caring for them,
using published research literature. Scope is decided by the **outcome** a question asks
about, not by the intervention it names.

## Areas

   - glycemic_control: blood glucose, HbA1c, glucose variability, hypoglycemia, insulin
     sensitivity or resistance, diabetes remission
   - body_composition_weight: body weight, fat distribution, lean or muscle mass, waist
     circumference, BMI
   - diet_nutrition: foods, nutrients, eating patterns, meal timing, alcohol, and
     supplements taken as part of diet
   - diet_medication_interaction: any food, drink, or eating pattern combined with any
     diabetes medication, including insulin

## Rule: outcome, not terms

A question is **in_charter** when its outcome maps to one of the four areas, whatever the
intervention is: a food, a drug, exercise, or surgery. Words like "insulin", "medication",
"drug", "exercise", or "surgery" never make a question adjacent on their own. When it is
uncertain whether a question is in charter or adjacent, it is in charter: the evidence check
that runs next still prevents unsupported answers.

## Population

Questions explicitly about type 1 or gestational diabetes are **adjacent**.

## Adjacent examples

eye, nerve, kidney, or foot complications; mental health; devices; cost or access to care

A question is **adjacent** only when it is about diabetes but its outcome maps to none of the
four areas. A question is **unrelated** when it is not about diabetes or metabolic health.

## Personal dosing (safety flag, not a scope category)

A question asking for a dose, schedule, or medication change for a specific person is still
classified by scope as usual, and is flagged. Flagged questions are answered from the evidence
with a banner telling the person to confirm any dose or medication change with their
clinician, and the answer gives no individualized dosing advice.

## Example questions

In charter:
- "Does a low-carbohydrate diet lower HbA1c in adults with type 2 diabetes?" (diet_nutrition)
- "Does exercising after meals lower post-meal blood glucose?" (glycemic_control; exercise is the intervention)
- "Is drinking alcohol risky while taking a sulfonylurea?" (diet_medication_interaction; alcohol is diet)

Adjacent:
- "How often should someone with diabetes get a dilated eye exam?" (eye complications)
- "Which continuous glucose monitor has the longest sensor life?" (devices)
- "What insulin-to-carbohydrate ratio should a person with type 1 diabetes use?" (type 1)

Unrelated:
- "What is the capital of Australia?"
- "How do I refinance a car loan?"
- "What's a good recipe for banana bread?"

## Notes for maintainers

- The four areas above must stay in sync with `src/ingest/config.py`'s `TOPICS` dict, which
  drives what gets ingested. The scope judge knows only what this charter says the system is
  meant to cover, not what the corpus contains.
- Revised 2026-09-28: moved from a topic-based charter (which listed exercise programming and
  insulin dosing as adjacent) to the outcome-based rule above, together with the rewritten
  Stage A prompt.
