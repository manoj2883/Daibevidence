# DiabEvidence

A trustworthy retrieval-augmented QA system for **diabetes research questions** — type 1, type 2, gestational, and prediabetes. It answers strictly from PubMed evidence it has ingested — every claim is cited inline by PMID, it states which population the evidence applies to (and flags it when that differs from what was asked), and it declines to answer when the retrieved evidence doesn't support one. It is **not medical advice**, and medication-related answers are informational only — never dosing or prescribing guidance.

The contribution is the trustworthy retrieval method, not diabetes answers specifically — diabetes is the test bed; the same pipeline works for any clinical domain by swapping the corpus. Direct Pinecone and Anthropic API calls only — **no LangChain** — for fewer dependencies and an easier-to-explain pipeline.

## Repository layout

```
src/ingest/config.py       # single source of truth for scope/filters/limits — edit this to change the corpus
src/ingest/pubmed.py       # NCBI E-utilities client: search, fetch, parse, classify, cache raw JSON/CSV
src/ingest/population.py   # keyword classifier: type1 / type2 / gestational / prediabetes / mixed
src/ingest/chunker.py      # word-based chunking + chunk CSV export
src/ingest/local_embeddings.py  # fastembed (ONNX) wrapper (shared by ingestion and retrieval)
src/ingest/uploader.py     # Pinecone index creation + idempotent upsert
src/ingest/freeze.py       # corpus freeze manifest (provenance + stats)
src/ingest/run_ingest.py   # ingestion pipeline entrypoint (CLI)
src/rag/types.py           # RetrievedChunk — plain chunk type shared by retrieval and generation
src/rag/cost.py            # prompt-size estimate + hard chunk-count safety cap before any Claude call
src/rag/retriever.py       # query embedding + direct Pinecone index.query()
src/rag/chain.py           # grounded, streaming generation: direct Anthropic SDK, citations, population-mismatch flag, disclaimer
src/rag/query_log.py       # appends every /query call + its retrieved chunks to data/query_log.jsonl
src/api/main.py            # FastAPI app: /query (streaming SSE), /health
src/api/index.html         # single-page chat frontend (vanilla HTML/CSS/JS, no build step)
eval/test_questions.json   # fixed 30-question in-scope evaluation set (see below)
eval/out_of_scope_questions.json  # unrelated conditions/trivia/nonsense, for tuning the similarity floor
scripts/tune_threshold.py  # empirically sweeps SIMILARITY_FLOOR — see data/threshold_sweep.csv
data/                      # gitignored local cache — except corpus_manifest.json (see "Freezing the corpus")
```

## Architecture

Two pipelines, both built on [Pinecone](https://www.pinecone.io/) as the vector store. Scope, filters, and limits are named constants at the top of `src/ingest/config.py` — that's the file to edit to change the corpus.

**Ingestion** (`python -m src.ingest.run_ingest`)
1. Query [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/) across 4 topics — diet & nutrition, glycemic control, body composition & weight, and diet/medication interaction — for up to `FETCH_LIMIT` (default 300) abstracts total, covering all diabetes types. Filtered to reviews, systematic reviews, meta-analyses, and clinical trials (`PUBLICATION_TYPES`); the last `DATE_RANGE_YEARS` (default 10) years; human studies only; English language.
2. Save the raw abstracts to `data/pubmed_raw.json` **and** `data/pubmed_raw.csv`, plus the exact queries/date-range used to `data/pubmed_fetch_meta.json` — all before anything else happens, so you can iterate on chunking/embedding without re-hitting PubMed (`--skip-fetch`), open the corpus in Excel, and know exactly what was asked of PubMed.
3. Chunk each abstract into `CHUNK_SIZE_WORDS`-word passages (default 300) with `CHUNK_OVERLAP_WORDS` overlap (default 50), tagging every chunk with `pmid` / `title` / `journal` / `year` / `publication_type` / `population` / `chunk_index`. `population` (`type1` / `type2` / `gestational` / `prediabetes` / `mixed`) is extracted from each abstract's text by a keyword classifier (`src/ingest/population.py`) during ingestion.
4. Save the processed chunks (with full metadata) to `data/pubmed_chunks.json` **and** `data/pubmed_chunks.csv` — again, before anything is sent to Pinecone.
5. Embed locally and for free with `sentence-transformers/all-MiniLM-L6-v2` (384-dim), run via [fastembed](https://github.com/qdrant/fastembed)'s pure-ONNX runtime rather than the PyTorch backend — no embedding API key or cost, and no torch/transformers dependency (~220MB RSS instead of 500MB+, which matters on Render's free 512MB tier).
6. Upsert into Pinecone (serverless, cosine metric, auto-created if the index doesn't exist yet) with a **stable ID per chunk** (`{pmid}_{chunk_index}`), so re-running the script updates existing vectors instead of duplicating them.
7. Write a **corpus freeze manifest** (`data/corpus_manifest.json`) — see "Freezing the corpus" below.

`data/` is gitignored — it's a regenerable local cache — with one deliberate exception: `data/corpus_manifest.json` is tracked, since it's the provenance record for whatever corpus is currently in Pinecone.

**Query** (`POST /query`, streams a server-sent-events response)
1. Embed the question with the same local model.
2. Retrieve a wide candidate pool (`RETRIEVAL_CANDIDATE_K`, default 10) from Pinecone, then apply a **similarity floor** (`SIMILARITY_FLOOR` env var, empirically tuned default 0.50 — see "Tuning the similarity floor" below) that decides how many of those candidates are actually relevant, instead of always forcing a fixed top-k. Nearest-neighbor search always returns *a* nearest neighbor, even when nothing in the corpus is relevant — the floor is what lets the system recognize that and abstain. Never sends more than `RETRIEVAL_CANDIDATE_K` chunks to Claude — a hard cap in `src/rag/cost.py` raises rather than silently sending more.
3. Three possible outcomes, exposed as `state` in the API response:
   - **`refused`** — nothing cleared the floor. A `refusal` event is sent with the closest scores found, a description of what the system does cover, and the full score distribution. **No Claude call is made.**
   - **`answered_low_confidence`** — something cleared the floor, but the best score is only marginally above it.
   - **`answered`** — a `sources` event (retrieved chunks + population info + a deterministic population-mismatch flag + the score distribution that produced the decision), then hand the surviving chunks to **Claude** (`messages.stream`, direct Anthropic SDK), which:
     - answers only from them, structured as one JSON object per sentence — see point 5 below
     - states which diabetes population the cited evidence applies to
     - explicitly flags a mismatch if the retrieved evidence covers a different population than the question asked about
     - discusses medications only in general informational terms — never dosing or prescribing advice
     - declines with the fixed refusal sentence if, despite clearing the floor, the excerpts still don't support an answer
4. Before its answer, the model first compares the surviving excerpts for genuine contradictions — conflicting findings on the same specific claim (not just different topics, and not a population difference, which stays a separate mismatch flag). It emits a JSON preamble (delimited by fixed markers, parsed server-side and never shown to the user) which becomes its own `contradictions` event, sent before any `sentence` events. When a contradiction is found, the model's answer must present both positions rather than silently picking one, and state what differs between the studies (population, design, duration, sample size) that could explain the disagreement.
5. The answer itself is a JSON array of `{sentence, chunk_ids, supported}` objects, not a prose blob — citation attribution is structural (which `[Excerpt N]` chunk_ids support this specific sentence), not `[PMID: ...]` text markup. Each array element is parsed off the stream as soon as its closing brace completes (`find_complete_json_objects`, pure/unit-tested incremental scanner — respects string escaping so braces inside sentence text don't confuse it) and sent as its own `sentence` event with the chunk_ids resolved to real PMIDs, so the frontend gets a progressive, sentence-by-sentence reveal instead of raw token-by-token text. Any sentence the model can't attribute to a retrieved chunk is either omitted entirely (if it would be an unsupported clinical claim) or marked `supported: false` (if it's pure transition/framing). A final `done` event carries the disclaimer and the **groundedness** summary — the share of sentences with at least one supporting chunk, computed from `chunk_ids` presence rather than the model's self-reported `supported` flag. This is the headline metric for the evaluation section.
6. Every call — question, state, the exact chunks retrieved (PMID, score, population), and the final answer — is appended to `data/query_log.jsonl`. This is the raw material for the evaluation section; it isn't reconstructable after the fact, so it's logged unconditionally rather than only during a formal eval run.

### Tuning the similarity floor

`SIMILARITY_FLOOR` is not guessed — `scripts/tune_threshold.py` retrieves the real candidate pool (Pinecone only, no Claude calls, so it costs nothing) for `eval/test_questions.json` (30 in-scope questions) and `eval/out_of_scope_questions.json` (20 deliberately unrelated questions: unrelated conditions, general trivia, nonsense), then sweeps a range of floor values against the cached scores and reports in-scope answer rate vs. out-of-scope refusal rate at each step. The full sweep is saved to `data/threshold_sweep.csv`. On this corpus, floors from 0.48–0.50 achieve 100% in-scope answer rate and 100% out-of-scope refusal rate; `0.50` is the current default. Re-run the sweep after any corpus or embedding-model change:

```bash
python -m scripts.tune_threshold                          # sweep 0.30–0.70 in steps of 0.01
python -m scripts.tune_threshold --start 0.4 --stop 0.6 --step 0.005
```

## Freezing the corpus

PubMed's result set for the same query shifts over time as new articles are indexed, so "re-run the ingestion query" is not a valid way to reproduce a prior corpus later. Two files exist for this:

- `data/pubmed_fetch_meta.json` — written at fetch time: the fetch timestamp, the exact per-topic queries sent to E-utilities, the date range, and the full PMID list.
- `data/corpus_manifest.json` — written at the end of every ingestion run (unless `--no-manifest`): the fetch info above, plus corpus-level stats (population/topic/publication-type breakdowns, chunk count), the embedding model + dimension, the Pinecone index name, and the current git commit. **This is the only `data/` file that isn't gitignored** — commit it once you're happy with a corpus, so the exact evidence base behind a set of results is on record for reviewers.

Current corpus (frozen 2026-09-02, see `data/corpus_manifest.json` for full detail): **264 abstracts / 319 chunks** — population split `type2: 166, mixed: 57, type1: 28, gestational: 9, prediabetes: 4`; topic split `diet_nutrition: 75, diet_medication_interaction: 69, glycemic_control: 61, body_composition_weight: 59`. Note the thin gestational/prediabetes coverage — that's expected from the topic queries as written, not a bug, and it's exactly the kind of gap the eval set below is designed to surface (via refusals or explicit population-mismatch flags, not silent extrapolation from type 2 evidence).

## Evaluation

`eval/test_questions.json` is a fixed set of 30 questions, written **before** any prompt or retrieval tuning against this corpus, so tuning can't unconsciously bend the pipeline toward passing them. It covers: per-topic questions for each population, population-mismatch traps (asking about a population the corpus doesn't actually have matching evidence for — the system must flag this, not paper over it), medication-dosing traps (must decline, not answer), and out-of-scope traps (must return the exact refusal sentence). Each entry has an `expected_behavior` rubric, not a ground-truth answer — grade against the rubric, and if a question exposes a real gap, fix the pipeline, not the question.

Run questions through `/query` (or `get_rag_chain()` directly) and cross-reference `data/query_log.jsonl` for the retrieval side of each answer.

## Setup

```bash
python -m venv venv
source venv/Scripts/activate  # or venv\Scripts\activate on Windows cmd
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` (all secrets — never committed, `.env` is gitignored):
- `ANTHROPIC_API_KEY` — [console.anthropic.com](https://console.anthropic.com/)
- `PINECONE_API_KEY` / `PINECONE_INDEX_NAME` — [app.pinecone.io](https://app.pinecone.io/)
- `PUBMED_EMAIL` (recommended by NCBI etiquette) and optionally `PUBMED_API_KEY` (raises the rate limit from 3 to 10 req/s)

No Gemini/OpenAI key is needed — embeddings run locally. Non-secret pipeline tunables (fetch limit, date range, topics, publication types, retrieval count) live in `src/ingest/config.py`, not `.env`.

## Running the ingestion pipeline

```bash
python -m src.ingest.run_ingest                    # fetch up to FETCH_LIMIT abstracts, chunk, embed, upsert, freeze
python -m src.ingest.run_ingest --skip-fetch        # reuse data/pubmed_raw.json, no PubMed calls
python -m src.ingest.run_ingest --dry-run           # fetch + chunk + export CSV/JSON, skip Pinecone
python -m src.ingest.run_ingest --fetch-limit 50    # override the corpus size for a quick test
python -m src.ingest.run_ingest --no-manifest       # skip writing/overwriting the freeze manifest
```

## Running the API

```bash
uvicorn src.api.main:app --reload
```

- `GET /health` — reports whether Claude and Pinecone are configured
- `POST /query` — `{"question": "..."}` → `text/event-stream` of `sources` / `contradictions` / `sentence` / `done` / `refusal` / `error` events (see "Query" above; also logged to `data/query_log.jsonl`)

There is no live ingestion endpoint — the frontend is chat-only, and ingestion is CLI-only (`python -m src.ingest.run_ingest`), per the offline ingestion pipeline above.

## Frontend

`src/api/index.html` — a single vanilla HTML/CSS/JS file, no build step, no external dependencies (system fonts, hand-written SVG icons). Built for patients, not clinicians, but designed to survive being demoed to a researcher — evidence is visible by default, not hidden behind a toggle.

- **Per-sentence answer rendering**: each `sentence` event becomes its own `<span>`, color-coded by the population of its first supporting source. Hovering or focusing (keyboard-accessible) a sentence highlights the matching source card(s) in the evidence panel. Unsupported sentences (no `chunk_ids`) render muted/italic with a dashed underline — visually distinct, never presented as equivalent to a cited claim.
- **Evidence panel**: beside the answer on desktop, below it on mobile (a per-exchange two-column row, not a single global sidebar) — source cards (title, journal, year, PMID link, population chip, similarity bar) are visible immediately, not behind a toggle.
- **Retrieval log**: a collapsible `<details>` inside the evidence panel — every candidate score with a cleared/dropped marker against the floor, plus the timing breakdown (retrieval / contradiction check / generation / total) from each event's `timing_ms`.
- **Confidence indicator**: a plain-language badge for `answered` ("Grounded in N sources") and `answered_low_confidence` ("Limited evidence — this answer may be less reliable"). A `refused` response gets a fully designed empty state — heading, the scope description, the closest scores found, and clickable example in-scope questions — not an error message. The same designed state also covers the case where the model itself declines mid-answer (a single unsupported sentence) even though the floor let some weak candidates through, so the two refusal paths (Task 1's floor and the model's own Rule 6 check) read identically to the user.
- **Contradiction display**: when the `contradictions` event is non-empty, each conflict renders as its own two-column panel — position A and position B side by side (stacked only below ~560px), with the "differs by" explanation underneath.
- **Design system**: a deliberately restrained base (off-white/near-black, a single deep-indigo accent for UI chrome, serif for reading content) with color spent in exactly one place — population-coded accents on sentences, source-card borders, and chips — so the evidence layer is the visually bold, memorable part of the page and everything else stays quiet.
- Light/dark mode (`prefers-color-scheme` + a manual toggle persisted in `localStorage`), responsive down to mobile, `prefers-reduced-motion` respected, and every interactive element is a real focusable control with a visible `:focus-visible` ring.

## Tests

```bash
pytest tests/ -v
```

Embedding, chunking, and population-classification tests run with no external dependencies. Tests that call the live Claude/Pinecone APIs are skipped unless `ANTHROPIC_API_KEY` is set.

## Deployment (Render)

`render.yaml` defines a free-tier web service. Set `ANTHROPIC_API_KEY`, `PINECONE_API_KEY`, and `PINECONE_INDEX_NAME` as secrets in the Render dashboard. Run the ingestion script locally (or as a one-off Render job) to populate Pinecone before querying — it isn't run automatically on deploy.

## Disclaimer

DiabEvidence summarizes published research literature. It is a research tool, not a source of medical advice, diagnosis, treatment, or medication guidance.
