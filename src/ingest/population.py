"""
Keyword-based diabetes population classifier, shared by ingestion (tags each
abstract) and the query pipeline (tags the user's question, to detect a
population mismatch against retrieved evidence).
"""
from src.ingest.config import POPULATION_KEYWORDS


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
