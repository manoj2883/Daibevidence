# DiabEvidence design decisions

One section per decision: what we tried, what failed, the evidence, and what changed. Every number
comes from a file in this repo, named next to it. "Dev set" means the 43 questions in
`eval/eval_set_v2.json` (24 in-scope, 17 diabetes-adjacent, 2 unrelated). Results on the dev set are
optimistic: several fixes below were designed by looking at failures on those same questions.

## 1. Deciding when to answer: similarity threshold → yes/no judge → two-stage judge → no scope judge

### 1a. Similarity threshold
**Tried.** Refuse when the best retrieved chunk's cosine similarity is below a floor (0.50).
**Failed.** Similarity measures whether a passage is *about diabetes*, not whether it *answers the
question*. 16 of 17 diabetes-adjacent questions (retinopathy, insulin pumps, neuropathy) scored at
or above the weakest in-scope question (top-1 0.5259). No floor separates them. A local
cross-encoder reranker did not fix it either (14 of 17 still overlapped).
**Evidence.** `docs/findings-retrieval-floor.md`; `data/eval_v1_retrieval_only_v2_5k.json`;
`data/eval_v1_retrieval_only_rerank_v2_5k.json`. At the 0.50 floor, 19/24 in-scope and 8/17 adjacent
questions were answered (`eval/REPORT_twostage_judge.md` §2).
**Changed.** The floor no longer decides anything; its scores are only logged.

### 1b. One yes/no answerability judge
**Tried.** A small model (Haiku) reads the question and the top 5 chunks and returns
`answerable: true/false` (`src/rag/judge.py`).
**Failed.** A single yes/no mixes two questions: "is this in our subject area?" and "do these
passages actually state the answer?". It accepted on topical overlap: ids 1, 10 and 22 were judged
answerable and produced UNSUPPORTED answers.
**Evidence.** `eval/REPORT_twostage_judge.md` §1–2: 13/24 in-scope and 5/17 adjacent answered.
**Changed.** Split into two judges.

### 1c. Two-stage judge (scope, then evidence)
**Tried.** Stage A decides scope from the question alone (`in_charter` / `adjacent` / `unrelated`,
`src/rag/scope_judge.py`). Stage B audits each claim in the question against each passage:
DIRECT / PARTIAL / ABSENT (`src/rag/evidence_judge.py`).
**Failed.** Stage A was the bottleneck. With the original prompts it labelled 7 of 24 in-scope
questions "adjacent" and refused them before the evidence was ever checked: 10/24 in-scope answered.
Rewritten prompts raised that to 14/24 on the dev set but only 3/8 on held-out questions, and
adjacent questions were almost never answered (1/17 dev, 0/5 held-out), even when the library held
good evidence for them.
**Evidence.** `eval/REPORT_twostage_judge.md` §2–3; `eval/results/final_2026-09-28/README.md`;
`data/eval_v3_revised_prompts_v2_5k.json` (14/24 in-scope, 1/17 adjacent, 0/2 unrelated; faithfulness
7 SUPPORTED, 5 PARTIALLY_SUPPORTED, 3 UNSUPPORTED of 15 answered).
**Changed (v3).** The scope judge is switched off (`SCOPE_JUDGE_ENABLED=false`; code kept). The
evidence judge alone decides: sufficient → `answered`, partial → `answered_partial`,
insufficient → `not_covered` ("Our research library doesn't cover this question yet."). An
unrelated question simply finds no evidence. The one safety job Stage A also did, spotting
personal dosing questions, moves to the query step (section 5), which routes them to
`see_clinician` with no generated answer. Result: section 9.

## 2. Chunking: word window → recursive chunking

**Tried.** v1_300 used a 300-word sliding window with 50 words of overlap. v2_5k kept the window but
stopped it from crossing abstract sections (`split_pubmed_documents_sectioned` in
`src/ingest/chunker.py`).
**Failed.** The window cuts mid-sentence, and chunks were longer than the embedding model reads.
all-MiniLM-L6-v2 was trained on 256 tokens, and our embedder (fastembed) actually stopped at
**128** tokens, read from the model's tokenizer config. 40.3% of the 13,733 v2_5k chunks are longer
than 128 tokens and 18.1% are longer than 256 (max 705, median 128), so the end of those chunks never
reached the vector.
**Evidence.** Measured with the model's own tokenizer over `data/v2_5k_chunks.json` (this upgrade, §1
check); `src/ingest/local_embeddings.py` docstring.
**Changed.** `src/ingest/recursive_chunker.py` splits by abstract sections, then sentences, then
words (only for a single sentence too long to fit). It never cuts mid-sentence otherwise, carries the
last full sentence forward as overlap, and limits header + chunk to 250 MiniLM tokens. The embedder
now reads up to 256 tokens. Every chunk starts with "Title | Year | Population | Study design".
Abstracts tagged "Retracted Publication" are skipped (2 in this corpus). Result: namespace
`v3_5k_recursive`, 14,658 chunks (14,630 PubMed + 28 background), none over 250 tokens
(`data/corpus_manifest_v3_5k_recursive.json`). v1_300 and v2_5k are untouched.

## 3. Parent-child retrieval

**Tried.** Sending the matched chunks themselves to the judge and to generation.
**Why change.** A chunk can hold the methods without the result, or the result without the
population, so a judge reading chunks may see only half of what a paper states.
**Evidence.** Not measured on its own. Run A (chunks) vs Run B (parents, plus the other v3 changes)
in section 9 measures it only in combination.
**Changed.** Small chunks are searched; the full parent abstract (or full background page) is what the
judge and generation read. Results are de-duplicated by PMID so one paper can't fill several slots.
Top 6 parents.

## 4. Hybrid search

**Tried.** Dense search only (embedding similarity).
**Why change.** Embeddings blur exact names: a question naming a specific drug, diet, or
measurement can retrieve passages about the general topic instead. Keyword search matches the
names exactly.
**Evidence.** Not measured on its own; part of Run B in section 9.
**Changed.** `src/rag/hybrid_retriever.py`: dense top 20 (on the rewritten query) plus BM25 top 20
(local `rank_bm25`, on the query step's keywords), merged with reciprocal rank fusion
(score = Σ 1/(60 + rank)). Both PubMed chunks and background pages are searched on every question.
Before, a regex sent each question to only one of the two (`src/rag/query_classifier.py`).

## 5. Query rewriting

**Tried.** Searching with the user's own words.
**Why change.** People write "high sugar in the morning"; abstracts write "morning hyperglycemia".
Plain-language questions land far from research vocabulary.
**Evidence.** Not measured on its own; part of Run B in section 9.
**Changed.** One Haiku call (`src/rag/query_rewriter.py`) returns `search_query` (research
vocabulary), up to 5 `keywords` (synonyms, specific names) and `personal_dosing`. It replaces the
scope judge call, so the number of model calls per question is unchanged. These fields are internal:
the evidence judge and the answer always get the original question, and the fields appear only in
the retrieval inspector (tested in `tests/test_v3_pipeline.py`).

## 6. Passage ordering

**Tried.** Passages in rank order, best first.
**Why change.** Language models attend most to the start and end of a long context and least to
the middle ("lost in the middle"). With full abstracts the context is several times longer.
**Evidence.** Not measured on its own; part of Run B in section 9.
**Changed.** Strongest passage first, second-strongest last, weaker ones in between
(`order_passages` in `src/rag/hybrid_retriever.py`).

## 7. Evidence badges

**Tried.** Source cards showed a similarity bar.
**Why change.** Similarity says nothing about how strong a study is, and after rank fusion there is
no single similarity score per source to show.
**Evidence.** Publication types stored per abstract in `data/v2_5k_raw.json` (e.g. 783 meta-analyses,
1,534 RCTs, 2,051 reviews among 5,000).
**Changed.** Each source card shows a badge from its PubMed publication type: Meta-analysis / RCT /
Observational / Review / Background (publisher). Trials PubMed doesn't tag as randomized (phase I–IV,
non-randomized) get a separate "Clinical trial" badge rather than being mislabelled. The badge also
goes into each chunk's header and the generation prompt.

## 8. Patient and clinician modes

**Tried.** One answer style for everyone (plain language at about an 8th-grade level, with drug
names and statistics from the studies).
**Why change.** One style can't serve both readers. For patients, naming drugs they never asked
about invites them to ask for, or switch to, a specific medicine.
**Evidence.** Patient-mode drug mentions are audited in section 9.
**Changed.** A header toggle (Patient by default, remembered in the browser). Only the generation
prompt changes; the evidence judge is identical in both modes. Patient mode: plain English, short
sentences, every medical term explained, no drug, drug class or supplement the user didn't name
("some diabetes medicines" instead). Any patient answer that touches medication ends with "Talk to
your doctor or pharmacist before changing anything." (added by code, not left to the model).
Clinician mode: study design, size, effect sizes, drug names and PMIDs up front. Hover definitions
for terms like HbA1c come from `config/glossary.json`.

## 9. Result

Dev set, patient mode, 2026-10-02. Files in `eval/results/v3_2026-10-02/`; `python -m scripts.report_v3`
rebuilds the table from `data/`.

| Config | In-scope answered /24 | Adjacent answered /17 | Unrelated /2 | Fully supported % |
|---|---|---|---|---|
| Baseline: two-stage judge, revised prompts | 14/24 | 1/17 | 0/2 | 47% (7/15) |
| Run A: v2_5k + background, no scope judge | 14/24 | 7/17 | 0/2 | 48% (10/21) |
| Run B: full v3 | 17/24 | 8/17 | 0/2 | 56% (14/25) |
| Run B + drug-name check (ids 8, 12 re-run) | 17/24 | 8/17 | 0/2 | 56% (14/25) |

- Dropping the scope judge (Run A) kept in-scope answers level and answered 6 more adjacent
  questions, with fewer UNSUPPORTED answers (1 of 21 vs 3 of 15).
- The full v3 pipeline (Run B) answered 3 more in-scope questions than either and had the highest
  share fully supported: 12 of 17 in-scope answers SUPPORTED; adjacent answers were mostly partial
  (2 SUPPORTED, 5 PARTIALLY_SUPPORTED, 1 UNSUPPORTED of 8). UNSUPPORTED overall: ids 7, 12, 15, 35.
- Patient mode named an unmentioned medicine twice in Run B: id 8 ("basal insulin") and id 12
  ("metformin", "SGLT-2 inhibitors"); the prompt rule alone does not hold. Added a deterministic check
  (`src/rag/safety.py` lexicon, `_patient_generation` in `src/rag/pipeline.py`): scan the answer,
  regenerate once with the flagged names forbidden, then replace any left with "some diabetes
  medicines". Catches are logged to `data/drug_check_log.jsonl`. Re-running ids 8 and 12: both caught,
  both fixed by the regeneration (no replacement needed). Id 8 stayed SUPPORTED; id 12 went from
  UNSUPPORTED to PARTIALLY_SUPPORTED. Patient answers now hold back text until the check passes.
- Smoke test (unscored): type 1 vs type 2 → answered_partial (2 PubMed reviews + 3 ADA/CDC pages);
  "What is HbA1c?" and "What is insulin resistance?" → answered, from PubMed reviews/meta-analyses
  only; no background page reached the top 6.
- Questions refused by the baseline and answered by Run B: in-scope ids 12, 15, 17, 19, 22; adjacent
  ids 25, 31, 33, 34, 35, 36, 37. Of these, ids 15, 19, 22, 25, 33, 34, 36, 37 cite at least one
  passage found only by BM25 (not in dense top 20), and every one except id 17 (and id 35, which
  cites nothing) cites a paper that Run A's dense top-6 chunks did not contain. Ids 12, 15, 35 were UNSUPPORTED; id 35 cites no passage at all.
  Detail per id: `eval/results/v3_2026-10-02/moved_to_answered.md`.
- A parser bug dropped answers the model wrote before its contradiction block (3 answers in each
  run). Fixed in `src/rag/chain.py`; only those 6 questions were re-run. The invalid outputs are kept
  as `*.INVALID_empty_answers.*`.

## 10. Post-generation answer checks

**Why change.** Run B's 25 answered cases split sharply by status: `answered` was 11/12 fully
supported (92%), `answered_partial` 3/13 (23%), and all 3 UNSUPPORTED answers were partial; one
(id 35) cited no passage at all.
**Changed.** `src/rag/answer_checks.py`, run on every answer before it is shown, no model calls:
a citation outside the retrieved passages (or a PMID in the text that wasn't retrieved) →
`not_covered`, logged as a bug; an uncited factual sentence → removed, answer becomes
`answered_partial`, logged; no citations left → `not_covered`. Sentences about the evidence itself
(what the studies don't show, which population they cover) are kept. Answers in both modes are now
held back until the checks pass. Log: `data/answer_check_log.jsonl`.
**Result** (25 answered ids re-run, `eval/results/v3_2026-10-02/eval_v3_runB_checks_report.md`):

| | answered | answered_partial | all |
|---|---|---|---|
| Before | 11/12 (92%) | 3/13 (23%) | 14/25 (56%) |
| After | 8/10 (80%) | 7/15 (47%) | 15/25 (60%) |

- The checks removed 8 sentences in 8 answers. 6 were real uncited claims (ids 1, 5, 8, 10, 16).
  2 were wrong removals of sentences about the evidence (ids 19, 33); the pattern was fixed and
  tested afterwards, without re-running those ids.
- No answer was refused: none had zero citations or a citation outside its passages, and no bug
  was logged. Id 35 cited passages this time.
- The re-run regenerated every answer, so some grade changes are run-to-run variation, not the
  checks: ids 12 and 36 dropped to UNSUPPORTED and ids 7 and 15 rose, with the check reporting "ok".
