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

# --- File paths -----------------------------------------------------------

RAW_JSON_PATH = "data/pubmed_raw.json"
RAW_CSV_PATH = "data/pubmed_raw.csv"
CHUNKS_JSON_PATH = "data/pubmed_chunks.json"
CHUNKS_CSV_PATH = "data/pubmed_chunks.csv"
FETCH_META_PATH = "data/pubmed_fetch_meta.json"
CORPUS_MANIFEST_PATH = "data/corpus_manifest.json"
QUERY_LOG_PATH = "data/query_log.jsonl"

# --- Chunking ---------------------------------------------------------

CHUNK_SIZE_WORDS = 300
CHUNK_OVERLAP_WORDS = 50
