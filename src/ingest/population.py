"""
Keyword-based diabetes population classifier, shared by ingestion (tags each
abstract) and the query pipeline (tags the user's question, to detect a
population mismatch against retrieved evidence).
"""
import re
from typing import List, Set, Tuple

from src.ingest.config import POPULATION_KEYWORDS, POPULATION_MESH_MAP

# The POPULATION_KEYWORDS phrases (e.g. "type 1 diabetes") require that exact
# contiguous phrase, so a compound mention like "type 1 and type 2 diabetes"
# only matches type2 — "type 1" isn't followed by "diabetes" there, "type 2"
# is — and type1 silently drops out. These patterns catch the short "type 1"
# / "type 2" form independently of what immediately follows, so both are
# recognized in a compound mention. Gated on "diabet" appearing anywhere in
# the text (checked by the caller) so a bare "type 1"/"type 2" doesn't fire
# on an unrelated statistical "type 1 error" mention. The negative lookahead
# on "type i" keeps it from matching the "i" inside "type ii".
_TYPE1_SHORT_PATTERN = re.compile(r"\btype\s*-?\s*1\b|\btype\s*i\b(?!i)|\bt1dm\b", re.IGNORECASE)
_TYPE2_SHORT_PATTERN = re.compile(r"\btype\s*-?\s*2\b|\btype\s*ii\b|\bt2dm\b", re.IGNORECASE)


def classify_population(text: str) -> Set[str]:
    """
    Returns the set of populations actually named in `text` — one or more of
    "type1", "type2", "gestational", "prediabetes" — or {"mixed"} if none are
    named (covers both "genuinely unspecified" and "nothing recognized").
    Unlike the single-value convention document tagging still uses (see
    classify_population_single), a question that names two populations keeps
    both rather than collapsing to "mixed": collapsing was the actual bug
    behind "What is type 1 and type 2 diabetes" being tagged type2-only and
    triggering a spurious population-mismatch warning (the phrase-based
    keyword match below only catches "type 2 diabetes" there; the regex pass
    catches "type 1" independently of what follows it).
    """
    text_lower = (text or "").lower()
    matched = {
        population
        for population, keywords in POPULATION_KEYWORDS.items()
        if any(keyword in text_lower for keyword in keywords)
    }
    if "diabet" in text_lower:
        if _TYPE1_SHORT_PATTERN.search(text_lower):
            matched.add("type1")
        if _TYPE2_SHORT_PATTERN.search(text_lower):
            matched.add("type2")
    return matched or {"mixed"}


def classify_population_single(text: str) -> str:
    """
    Collapses classify_population()'s set to the single-tag convention
    document/chunk-level tagging still uses ("mixed" when more than one
    population is mentioned, or none) — ingestion tags one document with one
    summary population tag; it doesn't need the multi-population awareness
    the query-time fix above added, and changing the stored per-chunk tag
    convention would require re-tagging the whole already-uploaded corpus
    for no behavioral benefit at ingestion time.
    """
    matched = classify_population(text)
    return matched.pop() if len(matched) == 1 else "mixed"


def classify_population_two_pass(mesh_terms: List[str], text: str) -> Tuple[str, str]:
    """
    Phase 3: pass 1 checks the article's own MeSH descriptor headings
    (expert-assigned at indexing time) against POPULATION_MESH_MAP — a
    match there is authoritative, so it's tagged confidence "high". Only
    when no such MeSH descriptor is present does this fall back to the
    same free-text keyword heuristic classify_population_single() already
    uses elsewhere, tagged confidence "low" since it's a weaker signal.

    Returns (population, confidence) — unchanged single-tag contract, see
    classify_population_single.
    """
    mesh_lower = {m.strip().lower() for m in mesh_terms if m}
    matched = {POPULATION_MESH_MAP[m] for m in mesh_lower if m in POPULATION_MESH_MAP}

    if len(matched) == 1:
        return matched.pop(), "high"
    if len(matched) > 1:
        return "mixed", "high"

    return classify_population_single(text), "low"
