"""
Keyword-based diabetes population classifier, shared by ingestion (tags each
abstract) and the query pipeline (tags the user's question, to detect a
population mismatch against retrieved evidence).
"""
from typing import List, Tuple

from src.ingest.config import POPULATION_KEYWORDS, POPULATION_MESH_MAP


def classify_population(text: str) -> str:
    """
    Returns one of "type1", "type2", "gestational", "prediabetes", or "mixed".
    "mixed" covers both "multiple populations mentioned" and "none detected/unclear".
    """
    text_lower = (text or "").lower()
    matched = {
        population
        for population, keywords in POPULATION_KEYWORDS.items()
        if any(keyword in text_lower for keyword in keywords)
    }
    return matched.pop() if len(matched) == 1 else "mixed"


def classify_population_two_pass(mesh_terms: List[str], text: str) -> Tuple[str, str]:
    """
    Phase 3: pass 1 checks the article's own MeSH descriptor headings
    (expert-assigned at indexing time) against POPULATION_MESH_MAP — a
    match there is authoritative, so it's tagged confidence "high". Only
    when no such MeSH descriptor is present does this fall back to the
    same free-text keyword heuristic classify_population() already uses
    elsewhere, tagged confidence "low" since it's a weaker signal.

    Returns (population, confidence).
    """
    mesh_lower = {m.strip().lower() for m in mesh_terms if m}
    matched = {POPULATION_MESH_MAP[m] for m in mesh_lower if m in POPULATION_MESH_MAP}

    if len(matched) == 1:
        return matched.pop(), "high"
    if len(matched) > 1:
        return "mixed", "high"

    return classify_population(text), "low"
