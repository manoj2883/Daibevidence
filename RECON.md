# Recon: current pipeline, before the two-stage judge rebuild

Written before any code changes for this task. Covers exactly what was asked; flags in
**Conflict** notes anywhere this prompt's assumptions don't quite match what's in the repo.

## 1. The current judge

**File:** `src/rag/judge.py` (117 lines).

**Model:** `get_judge_model()` returns `os.environ.get("ANTHROPIC_JUDGE_MODEL", "claude-haiku-4-5")`.
**Generation model** (for later reuse): `src/rag/chain.py`'s `get_model()` returns
`os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")`. These two getters are the only model
strings in the repo's RAG path — the rebuild reuses both, introducing no new model name.

**Prompt** (`JUDGE_SYSTEM_PROMPT`): a single system prompt asking "do these excerpts contain
enough evidence to actually answer THIS specific question," with one worked example
(retinopathy excerpt not answering a diet/glycemic-control question), returning
`{"answerable": true|false, "reason": "<one sentence>"}`.

**Parsing:** `_strip_code_fences()` strips a ` ```json ` wrapper if present; `_parse_judge_response()`
`json.loads`s the result, validates `answerable` is a `bool` and `reason` is a non-empty `str`,
and **fails closed** (`answerable: False`, error logged) on any JSON error, non-dict response,
or missing/wrong-typed `answerable`. No retry — a single attempt, then fail closed. This
prompt's Step 4 wants "retry once, then fail closed" — a slightly stricter contract than the
existing judge, which the new Stage B judge will match; the existing boolean judge is being
replaced, not extended, so its own retry behavior doesn't need to change retroactively.

**Where consumed:** `src/rag/chain.py`'s `stream_answer()`, between retrieval and generation
(see §2). The judge is only called on chunks retrieval actually returned (`chunks[:JUDGE_CHUNK_LIMIT]`,
`JUDGE_CHUNK_LIMIT = 5`) — it never sees zero chunks (that case is handled by a separate
defensive branch before the judge is ever invoked, see §2).

**Conflict:** the prompt says "the boolean conflates scope and evidence." That's accurate,
but note the *current* judge already only fires after a chunk-retrieval step that itself
already applies population/tier routing (see §3) — the conflation is specifically "is this
in the corpus's 4 topics" vs. "do these 5 chunks answer this specific question," not a
retrieval-vs-relevance conflation.

## 2. Full request path and every status value

`POST /query` (`src/api/main.py`) → `stream_answer()` (`src/rag/chain.py`), an SSE generator.
Every state is set at exactly one of these points:

| Step | Function | What happens |
|---|---|---|
| 1 | `classify_population()` (`src/ingest/population.py`) | Tags the question with a population set (`type1`/`type2`/`gestational`/`prediabetes`/`mixed`) — keyword heuristic, no LLM call |
| 2 | `classify_and_decompose()` (`src/rag/query_classifier.py`) | Splits compound questions, tags each sub-question `definitional` (→ background tier) or `evidence_seeking` (→ study tier) — regex heuristic, no LLM call |
| 3 | `_retrieve_for_question()` → `retrieve()` (`src/rag/retriever.py`) | Pinecone query per sub-question, tier-filtered. Pulls `RETRIEVAL_CANDIDATE_K=10` candidates, takes top `RETRIEVER_TOP_K=5` by raw score (regardless of the floor — see §3) |
| 4 | `stream_answer()`, defensive branch | **If literally nothing was retrieved** (e.g. empty namespace) → `state="out_of_scope"`, `refusal` event, judge never called. This is the *only* remaining place `out_of_scope` is set today outside the judge |
| 5 | `judge_answerability()` | If `answerable: false` → `state="out_of_scope"`, `refusal` event, **no generation call** |
| 6 | `client.messages.stream(model=get_model(), ...)` | Generation, using `SYSTEM_PROMPT`'s 10 numbered rules (grounding-only, tier restriction, structured per-sentence JSON, contradiction preamble, population-mismatch flag, rule 7's "say so if no excerpt answers the specific claim") |
| 7 | `determine_final_state()` | **Post-generation**, the authoritative state: `no_evidence_for_claim` if no evidence-tier chunk was actually cited by any sentence; else `answered_low_confidence` if the top cited evidence chunk's score is within `LOW_CONFIDENCE_MARGIN` (0.05) of the floor; else `answered` |

So today's full enum is `answered` / `answered_low_confidence` / `no_evidence_for_claim` /
`out_of_scope`, set at steps 4, 5, or 7 as above. `_provisional_state()` computes a cheap
pre-generation guess of the same enum for the `sources` event (usually right, corrected by
step 7 in `done`).

**SSE events:** `sources` (once, pre-generation: chunks, population info, judge verdict,
score distribution), `contradictions` (once, pre-sentence), `sentence` (one per parsed
sentence), `done` (final state + groundedness + usage + model string), `refusal` (steps 4/5,
zero-generation-cost), `error` (misconfiguration or generation failure).

## 3. Similarity-floor logic — still present, no longer gates anything

`src/rag/retriever.py`: `decide_retrieval_state()` (pure function) and `retrieve_with_floor()`
(wraps it with a live Pinecone call) still exist, unchanged, and are still called by
`_retrieve_for_question()` on every request — **but only to compute `candidate_scores`/`floor`/
`background_floor`/`low_confidence_margin` for the retrieval-inspector payload
(`score_distribution` in the `sources`/`refusal` events).** Chunk *selection* for the
judge/generation is `candidates[:get_retriever_k()]` — top-5 by raw score, unconditional on
the floor. `SIMILARITY_FLOOR_DEFAULT = 0.50` (`src/ingest/config.py`), overridable via
`SIMILARITY_FLOOR`/`SIMILARITY_FLOOR_BACKGROUND` env vars.

`_retrieve_for_question()` also has a keyword-only `retrieval_mode` param (`"topk"` default,
`"floor"` for floor-gated selection) used exclusively by `scripts/run_old_pipeline_full.py` to
reconstruct the pre-judge pipeline for the last eval round's comparison — the live `/query`
path never passes it. `stream_answer()` similarly has a diagnostic-only `force_judge` param.

**This task's Step 5 asks to "retire ... any similarity-floor gating."** There already isn't
any in the live path — the floor only drives inspector display today. The two-stage rebuild
removes the floor from the *decision* path entirely (it was already out) and keeps the raw
`candidate_scores`/`top_score` numbers flowing into the response for the inspector, per this
task's instruction.

## 4. Eval runner, faithfulness grader, and the four data files

**`scripts/run_eval_v1.py`**: `run_one_question(q, namespace, **stream_kwargs)` runs the live
pipeline and captures full detail (sources, sentence_records, judge verdict, score
distribution, model strings, timing). `summarize()` computes per-topic and old-vs-new
comparison stats from a cached retrieval-only baseline. `scripts/run_old_pipeline_full.py`
and `scripts/build_judge_report.py` build on top of this.

**`scripts/grade_faithfulness.py`**: `GRADER_MODEL_DEFAULT = "claude-haiku-4-5"` —
**the same model as the judge** (`ANTHROPIC_GRADER_MODEL` env var, unset by default, so it
resolves to the same Haiku snapshot `ANTHROPIC_JUDGE_MODEL` does). `GRADER_SYSTEM_PROMPT` grades
each sentence SUPPORTED/PARTIALLY_SUPPORTED/UNSUPPORTED/NOT_APPLICABLE against its cited
excerpts, identifies a "central claim" sentence, and gives one `overall_verdict`. **This task's
Step 8 concern is valid**: same model, and it shares real phrasing with the judge/Stage-B
concept ("Topical relatedness/similarity is not evidence," "DIRECT/PARTIAL/ABSENT"-style
per-claim auditing was independently arrived at in both, but the grader was written after and
by the same author as the judge, so correlated blind spots are a real risk, not a hypothetical
one). Rewritten in Step 8 to (a) use `get_model()` (Sonnet — a different model already
configured in the repo, per the task's instruction) and (b) share no prompt phrasing with the
Stage B prompt.

**The four files, exactly as they are:**
- `data/eval_v1_report_figures.json` — every number the last report used, computed fresh each
  run by `build_judge_report.py` (contingency tables, genuine-regression/reclassified/
  would-have-failed-anyway lists, faithfulness distributions, model versions, latency).
- `data/eval_v1_summary_v2_5k_judge_v2.json` — the (old, boolean) judge-gated pipeline's full,
  real, end-to-end run over all 43 questions: per-question `final_state`, `judge_verdict`,
  `judge_reason`, `sources` (full excerpt text), `sentence_records`, `top_score`, timing, usage,
  `model_generation`.
- `data/eval_v1_old_pipeline_full_v2_5k.json` — the pre-judge (similarity-floor-gated) pipeline,
  same full detail, same 43 questions, for comparison.
- `data/eval_v1_faithfulness_v2_5k.json` — the faithfulness grader's verdicts for the specific
  answer instances the last report's reconciliation needed (not all 43 — see below).

**Conflict:** this task's Step 9 asks for "faithfulness grader on every answered case... not a
subset," and a "previous judge status" column implying the existing files already cover every
answered question. They don't: `eval_v1_faithfulness_v2_5k.json` only graded 24 of the ~32
answered instances across both pipelines (the ones needed for the specific regression/false-accept
reconciliation the last task asked for), not the full set. This is fine — Step 9's new run grades
every answered case fresh anyway — but the *previous* pipeline's per-question faithfulness in the
new comparison table will show "not previously graded" for ids that weren't in that 24, rather
than a real prior verdict. Noted so the table doesn't imply a gap that doesn't exist.

## 5. The frozen eval question file

`eval/eval_set_v2.json` — 43 questions, **not modified, will not be modified**. Schema per
question: `id`, `topic`, `in_scope` (bool), `question`, `reference_answer` (hand-written,
`null` where out-of-scope or no single consensus claim), and `sub_area` on the 17
diabetes-adjacent-off-topic questions only. `topic` values: 4 in-scope topics
(`diet_nutrition`, `glycemic_control`, `body_composition_weight`, `diet_medication_interaction`,
6 questions each), `out_of_scope_diabetes_adjacent` (17, six `sub_area`s: retinopathy,
pump_cgm_hardware, neuropathy, foot_care, dka, insurance_licensing), `out_of_scope_unrelated` (2).

**Conflict, informational:** this eval set's `in_scope`/`topic` labels are a coarser binary
than the new `scope_charter.md`'s three-way `in_charter`/`adjacent`/`unrelated` split — but
they line up exactly (`in_scope=true` ↔ `in_charter`, `out_of_scope_diabetes_adjacent` ↔
`adjacent`, `out_of_scope_unrelated` ↔ `unrelated`), so the confusion matrix in Step 9 is a
direct, unambiguous comparison, not an approximation.

## 6. Frontend status rendering

`src/api/index.html`, all vanilla JS, no framework. Current handling, by SSE event:

- `onSources`: renders the low-confidence notice (checks `data.state === "answered_low_confidence"`
  literally — the only place any status string is checked by name), the population-mismatch
  notice (checks `data.mismatch`, a *server-computed boolean*, independent of the model's own
  prose — see `is_population_mismatch()` in `chain.py`), the source-card list, and the retrieval
  inspector (`buildInspectorBody()` — score bars + floor line + judge verdict/reason block +
  raw JSON).
- `onDone`: if the answer degraded to exactly one unsupported sentence (the `no_evidence_for_claim`
  shape), re-renders `answerText` as a refusal block via `buildRefusalBlockHtml()` — **this is
  status-shape detection, not a literal state-string check**. Otherwise renders the "copy as
  markdown" action row.
- `onRefusal`: hides the evidence column, renders `buildRefusalBlockHtml(message, scope_description,
  closest_scores, judge)` — the judge block shows the reason as the refusal's evidentiary
  explanation, replacing the "closest scores ... below the floor" framing when a real (non-null)
  judge verdict is present.
- `onError`: plain error banner.

No status besides `answered_low_confidence` is checked by literal string anywhere in the
frontend today — everything else is inferred from event type + payload shape. The rebuild adds
literal checks for the four new statuses (`answered_partial`, `in_scope_no_evidence`,
`answered_adjacent`, and the retained `out_of_scope`), since this task's Step 6 wants each
rendered distinctly rather than inferred.

## 7. Id 20 — cause found (classifier bug, not retrieval/chunking/metadata)

Question: *"What is the interaction risk between alcohol intake and sulfonylurea medications?"*
(`diet_medication_interaction`, in-scope). `classify_question_type()`
(`src/rag/query_classifier.py`) matches this against `DEFINITIONAL_LEAD_PATTERN` (`^what\s+is`)
and finds no match in `EVIDENCE_SEEKING_CUE_PATTERN` (no "study/trial/evidence/does X
improve/..." wording — "interaction" isn't in the cue list), so it falls through to
`"definitional"` — which routes retrieval to the **background tier only** (35 generic
ADA/NIDDK patient-education chunks; no drug-interaction content). Confirmed live: the same
question against the **study tier** returns a directly relevant paper at 0.514
("Factors of primary and secondary sulfonylurea failure...") plus several wine/alcohol papers
in the 0.48–0.50 range; against the background tier only, top score is 0.298 (generic "About
Type 2 Diabetes" chunk). `decompose_compound()` correctly does *not* split this question (the
"and" joins two nouns — "alcohol intake" and "sulfonylurea medications" — not two clauses), so
it's classified as a single unit and the "what is" lead wins by default.

This is a **code bug in `query_classifier.py`**, not a retrieval, chunking, or metadata problem
— fixed in this change (§Step 7 below) by adding an interaction/drug-interaction cue to
`EVIDENCE_SEEKING_CUE_PATTERN` so "what is the interaction/risk between X and Y" routes to the
study tier like the evidence-seeking question it actually is, without touching
`decompose_compound()` or the definitional-lead pattern (which are both correct and needed for
genuinely definitional leads like "What is the dawn phenomenon").
