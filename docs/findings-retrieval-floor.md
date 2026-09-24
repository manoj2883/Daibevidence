# Finding: retrieval similarity cannot gate in-domain, off-corpus questions

**Date:** 2026-09-16 (n=4 slice) / 2026-09-16 (n=17 expansion, cross-encoder) · **Corrected:** 2026-09-18
**Namespace:** v2_5k (5,000 abstracts, 13,733 study-tier chunks, Pinecone index `diabevidence`)
**Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, local (fastembed/ONNX)
**Source data:** `data/eval_v1_retrieval_only_v2_5k.json` (n=4 slice), `data/eval_v1_retrieval_only_rerank_v2_5k.json` (n=17 slice + cross-encoder)
**Cost:** none — Pinecone and local embeddings only, no generation calls

> **This document did not previously exist in the repository.** The finding below was written up once (2026-09-16) and only ever saved to a private cross-session memory note, never committed as a project doc — which is why a factual error in it (see Correction below) went uncaught: there was no reviewable file for anyone but the assistant to check. All numbers in this version were independently recomputed from the two JSON files above on 2026-09-18, not copied from the earlier write-up or from memory.

## Correction (2026-09-18)

The original write-up claimed *"three of the four adjacent off-topic questions (0.7157, 0.7041, 0.6351) score above the weakest in-scope question."* That is wrong. Recomputed directly from `data/eval_v1_retrieval_only_v2_5k.json`:

- Minimum in-scope top-1 score: **0.5259** (id 20, "alcohol/sulfonylurea interaction")
- All four adjacent-off-topic top-1 scores: **0.7157** (id 25), **0.7041** (id 27), **0.6351** (id 28), **0.5432** (id 26)
- **0.5432 (id 26) is also above 0.5259.** The correct count is **four of four**, not three of four.

This also invalidates a downstream claim made when the slice was later expanded to n=17 (below): the original write-up characterized the n=17 result ("16/17 = 94%") as containment getting *worse* than the n=4 baseline, using the wrong n=4 figure (3/4 = 75%) as the comparison point. With the corrected n=4 baseline (4/4 = **100%**), going to 94% at n=17 is actually a slight *decrease* in the contained proportion, not an increase — the min/max score range did widen at n=17, but "worse" as a headline characterization was wrong and is retracted below.

## Claim

A single global similarity floor over bi-encoder cosine scores separates diabetes content from non-diabetes content, and nothing finer. It cannot distinguish questions the corpus answers from questions that are merely about diabetes.

## Method

The frozen evaluation set was run through retrieval only, with no generation stage. Questions were grouped three ways before scores were inspected:

- **in-scope** (24) — covered by the four indexed topics
- **diabetes-adjacent, off-topic** (4, later expanded to 17) — about diabetes, outside the indexed topics
- **fully unrelated** (2) — no connection to diabetes

Top-1 (and top-5 mean) cosine similarity were read from the cached retrieval-only run. No retrieval was re-executed for the write-up; the numbers in this document were recomputed a second time, independently, directly from the cached JSON, on 2026-09-18.

## Result — n=4 adjacent-off-topic slice (`eval/eval_set_v1.json`)

| Group | n | Top-1 min | Top-1 max |
|---|---|---|---|
| In-scope | 24 | 0.5259 | 0.8618 |
| Diabetes-adjacent, off-topic | 4 | 0.5432 | 0.7157 |
| Fully unrelated | 2 | 0.2061 | 0.2907 |

The adjacent-off-topic range sits entirely inside the in-scope range — containment, not marginal overlap.

**All four** adjacent-off-topic questions score at or above the in-scope minimum (0.5259):

| Question (id) | Top-1 |
|---|---|
| Diabetic retinopathy laser treatment (25) | 0.7157 |
| Painful diabetic peripheral neuropathy treatment (27) | 0.7041 |
| Driver's-license medical-review rules (28) | 0.6351 |
| Insulin pump tubing mechanism (26) | 0.5432 |

Minimum in-scope score, for reference: id 20 ("alcohol/sulfonylurea interaction"), **0.5259**. The maximum off-topic score exceeds it by 0.19. **No threshold value separates the two groups** — any floor high enough to reject the retinopathy question (>0.7157) would also reject 14 of the 24 in-scope questions.

Mean top-5 shows the same pattern (adjacent max 0.6896 vs. in-scope min 0.5130 — see the original n=4 retrieval-only cache), so there is no alternative bi-encoder statistic to fall back on.

## Behavior at the current floor (`SIMILARITY_FLOOR_DEFAULT = 0.50`)

- 24/24 in-scope questions cleared the floor (0 over-refusals)
- 2/6 out-of-scope questions were refused pre-generation — both from the fully-unrelated pair
- 4/6 out-of-scope questions cleared the floor, all diabetes-adjacent

The floor is functioning as a topic detector (diabetes vs. not-diabetes), not a scope detector (the 4 indexed topics vs. diabetes-adjacent content).

## Follow-up: n=17 adjacent-off-topic slice (`eval/eval_set_v2.json`)

Expanded the diabetes-adjacent-off-topic slice from 4 to 17 questions (retinopathy, insulin pump/CGM hardware, peripheral neuropathy, diabetic foot care, DKA, insurance/licensing — 6 sub-areas), re-verified 2026-09-18 directly from `data/eval_v1_retrieval_only_rerank_v2_5k.json`:

| Group | n | Top-1 min | Top-1 max |
|---|---|---|---|
| In-scope | 24 | 0.5259 | 0.8618 |
| Diabetes-adjacent, off-topic | 17 | 0.4845 | 0.8295 |
| Fully unrelated | 2 | 0.2061 | 0.2907 |

**16 of 17 (94.1%)** adjacent-off-topic questions score at or above the in-scope minimum. Compared against the *corrected* n=4 baseline (4/4 = 100%), this is a modest **decrease** in the contained proportion, not the increase the uncorrected write-up claimed — though at both sample sizes the containment is overwhelming and the practical conclusion is unchanged.

## Follow-up: local cross-encoder reranking

Tried `src/rag/reranker.py` (fastembed ONNX, `Xenova/ms-marco-MiniLM-L-6-v2`, zero API cost) over the same cached top-k candidate pools, re-verified 2026-09-18:

| Group | n | Top-1 min | Top-1 max |
|---|---|---|---|
| In-scope | 24 | −4.1746 | 9.0456 |
| Diabetes-adjacent, off-topic | 17 | −7.0026 | 6.0169 |
| Fully unrelated | 2 | −10.9381 | −9.3406 |

**14 of 17 (82.4%)** adjacent-off-topic questions still score at or above the in-scope minimum under the cross-encoder — it does not separate in-scope from adjacent-off-topic either. It does cleanly separate the fully-unrelated pair from everything else, same as cosine — so it's at best a redundant domain gate, not an answerability gate.

## Interpretation

The bi-encoder (and the cross-encoder, tried as a fix) encode topical relatedness. For a corpus that is entirely about diabetes, the shared subject dominates the score, leaving little signal for whether a passage answers the specific question asked. Retrieval similarity measures *aboutness*, not *answerability* — and this holds at both n=4 and n=17, and under both a bi-encoder and a cross-encoder.

The practical consequence: the similarity floor (and a cross-encoder reranker layered on top of it) can only be justified as a coarse domain gate. Scope enforcement within the domain has to happen somewhere else — currently the generation stage (`src/rag/chain.py` rule 7 → `no_evidence_for_claim`), which requires a live Claude call.

## Limits of this finding

- No generation-stage results yet for the n=17 set — groundedness, citation attribution, and true refusal rate all require the full pipeline and remain outstanding (blocked on Anthropic API credit as of the original write-up; status should be re-checked before relying on this).
- Six (n=4 slice) / two (fully-unrelated) is still a small sample for the fully-unrelated group specifically — counts only, not a refusal-rate percentage.
- Only one cross-encoder model was tried (`Xenova/ms-marco-MiniLM-L-6-v2`). The claim "no retrieval-stage method can" separate the groups is not established for cross-encoders generally, only for this one.

## Next

1. Only run the generation-based evaluation, once, against the expanded (n=17) set, once Claude API access is confirmed available.
2. If gating diabetes-adjacent-off-topic content at retrieval time is still wanted, the next thing to try is a topic classifier scoped to the 4 indexed topics specifically (not a relevance-score threshold of any kind), not another embedding/reranking model — see Interpretation above for why a score-based approach is structurally unlikely to work here.
