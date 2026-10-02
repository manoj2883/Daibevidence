"""
Central, non-secret configuration for the diabetes evidence pipeline.
API keys and other secrets stay in .env — everything here is a tunable
that's safe to read straight out of source control.
"""

# --- Ingestion scope ---------------------------------------------------

# Max total abstracts to pull across all topics combined.
FETCH_LIMIT = 300

# Only include publications from the last N years.
DATE_RANGE_YEARS = 10

# PubMed Publication Type values to restrict results to.
PUBLICATION_TYPES = ["Review", "Systematic Review", "Meta-Analysis", "Clinical Trial"]

# Topic search fragments (E-utilities syntax), ANDed with DIABETES_SCOPE,
# the publication-type filter, and the humans/English/has-abstract filters.
# FETCH_LIMIT is split evenly across these.
TOPICS = {
    "diet_nutrition": (
        '("diet, food, and nutrition"[MeSH Terms] OR "nutrition therapy"[MeSH Terms] '
        'OR "diet therapy"[Subheading])'
    ),
    "glycemic_control": (
        '("glycemic control"[Title/Abstract] OR "blood glucose"[MeSH Terms] '
        'OR "hemoglobin a, glycosylated"[MeSH Terms])'
    ),
    "body_composition_weight": (
        '("body composition"[MeSH Terms] OR "body weight"[MeSH Terms] OR "weight loss"[MeSH Terms])'
    ),
    "diet_medication_interaction": (
        '(("diet, food, and nutrition"[MeSH Terms] OR "food-drug interactions"[MeSH Terms]) '
        'AND "hypoglycemic agents"[MeSH Terms])'
    ),
}

# All diabetes types in scope (broadened from type 2 only).
DIABETES_SCOPE = (
    '("diabetes mellitus"[MeSH Terms] OR "diabetes mellitus, type 1"[MeSH Terms] '
    'OR "diabetes mellitus, type 2"[MeSH Terms] OR "diabetes, gestational"[MeSH Terms] '
    'OR "prediabetic state"[MeSH Terms])'
)

# --- Retrieval / generation ---------------------------------------------

RETRIEVAL_TOP_K = 5

# --- Similarity floor / abstention (Task 1) ------------------------------
# Nearest-neighbor search always returns a nearest neighbor, even when
# nothing in the corpus is actually relevant. Instead of always sending a
# fixed top-k to the model, retrieve a wider candidate pool and let a
# similarity floor decide how many (if any) are relevant enough to use.

# How many candidates to pull from Pinecone before applying the floor.
RETRIEVAL_CANDIDATE_K = 10

# Below this cosine score, a chunk is dropped as irrelevant. This is a
# fallback default, not the final word — override with the SIMILARITY_FLOOR
# env var (set in Render's environment tab). Tuned empirically by
# scripts/tune_threshold.py against eval/test_questions.json (in-scope) and
# eval/out_of_scope_questions.json (unrelated conditions/trivia/nonsense);
# see data/threshold_sweep.csv for the sweep that produced this value.
SIMILARITY_FLOOR_DEFAULT = 0.50

# If the best surviving chunk's score is within this margin of the floor,
# mark the answer low-confidence instead of answering outright.
LOW_CONFIDENCE_MARGIN = 0.05

# --- Population classification ------------------------------------------
# Keyword heuristic used to tag each abstract (and each incoming question)
# with the diabetes population it concerns. Order-independent; a match on
# more than one group (or none) classifies as "mixed".
POPULATION_KEYWORDS = {
    "type1": [
        "type 1 diabetes", "type 1 diabetic", "t1dm", "type i diabetes",
        "insulin-dependent diabetes",
    ],
    "type2": [
        "type 2 diabetes", "type 2 diabetic", "t2dm", "type ii diabetes",
        "non-insulin-dependent diabetes", "noninsulin-dependent diabetes",
    ],
    "gestational": ["gestational diabetes", "gdm"],
    "prediabetes": [
        "prediabetes", "pre-diabetes", "impaired glucose tolerance", "impaired fasting glucose",
    ],
}

# Pass 1 of two-pass population tagging (Phase 3): PubMed's own MeSH
# indexing is a curated, expert-assigned signal — far more reliable than
# matching keywords in free text — so a MeSH descriptor match is tagged
# "high" confidence. Only falls back to the POPULATION_KEYWORDS heuristic
# above ("low" confidence) when none of these descriptors are present.
POPULATION_MESH_MAP = {
    "diabetes mellitus, type 1": "type1",
    "diabetes mellitus, type 2": "type2",
    "diabetes, gestational": "gestational",
    "prediabetic state": "prediabetes",
}

# --- File paths -----------------------------------------------------------

RAW_JSON_PATH = "data/pubmed_raw.json"
RAW_CSV_PATH = "data/pubmed_raw.csv"
CHUNKS_JSON_PATH = "data/pubmed_chunks.json"
CHUNKS_CSV_PATH = "data/pubmed_chunks.csv"
FETCH_META_PATH = "data/pubmed_fetch_meta.json"
CORPUS_MANIFEST_PATH = "data/corpus_manifest.json"
QUERY_LOG_PATH = "data/query_log.jsonl"

BACKGROUND_RAW_JSON_PATH = "data/background_raw.json"
BACKGROUND_RAW_CSV_PATH = "data/background_raw.csv"
BACKGROUND_CHUNKS_JSON_PATH = "data/background_chunks.json"
BACKGROUND_CHUNKS_CSV_PATH = "data/background_chunks.csv"

# --- Phase 3: corpus expansion (v2_5k) -------------------------------------
# Same TOPICS/DIABETES_SCOPE/PUBLICATION_TYPES/DATE_RANGE_YEARS as the
# original 300-abstract corpus (verified live against PubMed: ~25k matches
# available pre-dedup under this exact scope, comfortably more than 5k) —
# only FETCH_LIMIT changes, so v2_5k is a clean "same scope, more data"
# comparison against v1_300, not a scope change bundled in as well.
V2_5K_FETCH_LIMIT = 5000
V2_5K_RAW_JSON_PATH = "data/v2_5k_raw.json"
V2_5K_RAW_CSV_PATH = "data/v2_5k_raw.csv"
V2_5K_CHUNKS_JSON_PATH = "data/v2_5k_chunks.json"
V2_5K_CHUNKS_CSV_PATH = "data/v2_5k_chunks.csv"
V2_5K_FETCH_META_PATH = "data/v2_5k_fetch_meta.json"
V2_5K_MANIFEST_PATH = "data/corpus_manifest_v2_5k.json"
V2_5K_FETCH_CHECKPOINT_PATH = "data/v2_5k_fetch_checkpoint.json"
V1_300_MANIFEST_PATH = "data/corpus_manifest_v1_300.json"

# Pinecone namespaces for the Phase 3 side-by-side comparison. "" (the
# default namespace, untouched) stays the original live corpus.
NAMESPACE_V1_300 = "v1_300"
NAMESPACE_V2_5K = "v2_5k"

# --- Chunking ---------------------------------------------------------

CHUNK_SIZE_WORDS = 300
CHUNK_OVERLAP_WORDS = 50

# Background patient-education pages are already dense per paragraph — a
# smaller chunk size gives better retrieval granularity than reusing the
# evidence corpus's 300-word chunks (e.g. a "symptoms" question shouldn't
# have to pull in a whole page's "risk factors" and "management" text too).
BACKGROUND_CHUNK_SIZE_WORDS = 120
BACKGROUND_CHUNK_OVERLAP_WORDS = 20

# --- v3: recursive chunking + hybrid retrieval ------------------------------
# Same 5,000 abstracts as v2_5k (minus retractions) + background pages,
# re-chunked by src.ingest.recursive_chunker. corpus/ is tracked in git (data/
# is not) because the query path reads it at runtime for BM25 and parents.
NAMESPACE_V3 = "v3_5k_recursive"
V3_CORPUS_PATH = "corpus/v3_5k_recursive.json.gz"
V3_MANIFEST_PATH = "data/corpus_manifest_v3_5k_recursive.json"

# Hybrid retrieval: dense top-N + BM25 top-N, reciprocal rank fusion, dedupe
# by parent, keep the top FINAL parents.
V3_DENSE_TOP_K = 20
V3_BM25_TOP_K = 20
V3_RRF_K = 60
V3_FINAL_PARENTS = 6
