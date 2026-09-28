"""
Heuristic (no-LLM-call) question classification and decomposition. Every
question routed through stream_answer needs to know two things before
retrieval happens: is it asking for a definition/mechanism (route to the
background tier) or for what the research shows (route to the study/evidence
tier), and is it actually two questions stitched into one sentence that need
separate answers, in order?

Deliberately keyword/regex-based, same style as
src.ingest.population.classify_population, rather than an LLM call — this
keeps question routing free and instant, and (per the task this was written
for) avoids depending on Claude API availability for a decision that governs
which tier even gets queried.
"""
import re
from typing import List

DEFINITIONAL_LEAD_PATTERN = re.compile(
    r"^\s*(what\s+is|what\s+are|what\s+does\b.*\bmean|define|defines|definition\s+of|"
    r"how\s+is\b.*\bdiagnosed|how\s+are\b.*\bdiagnosed|how\s+does\b.*\bwork|"
    r"how\s+do\b.*\bwork|what\s+causes|what\s+is\s+the\s+difference|"
    r"what\s+are\s+the\s+symptoms|what\s+are\s+the\s+risk\s+factors)\b",
    re.IGNORECASE,
)

EVIDENCE_SEEKING_CUE_PATTERN = re.compile(
    r"\b(study|studies|trial|trials|evidence|research|rct|meta-analysis|"
    r"systematic review|does\b.*\b(improve|reduce|affect|help|increase|decrease|"
    r"cause|lower|raise|worsen)|is there evidence|what does the evidence show|"
    r"what does the evidence say|how effective|compared to|versus|vs\.|"
    # "What is the interaction/risk between X and Y" is an evidence-seeking
    # question about a relationship, not a request for a definition, even
    # though it starts with the generic "what is" lead (see RECON.md's
    # id-20 finding: this pattern was mis-tagged "definitional" and routed
    # to the background-only tier, missing real study-tier evidence).
    r"interaction\s+(risk|between|with)|drug\s+interaction|medication\s+interaction|"
    r"risk\s+between)\b",
    re.IGNORECASE,
)

# A clause after a splitting conjunction only counts as a *separate question*
# (triggering decomposition) if it starts with its own interrogative/auxiliary
# lead word — otherwise the conjunction is just joining two nouns within one
# question ("type 1 and type 2 diabetes" is one question about two entities,
# not two questions).
_NEW_CLAUSE_LEAD_WORDS = (
    "what", "how", "does", "do", "is", "are", "can", "should", "why", "when",
    "which", "will", "would", "could",
)
_NEW_CLAUSE_LEAD_PATTERN = re.compile(
    r"^\s*(" + "|".join(_NEW_CLAUSE_LEAD_WORDS) + r")\b", re.IGNORECASE
)

_SPLIT_PATTERN = re.compile(r"\s+and\s+|\s*;\s*|\s+but\s+", re.IGNORECASE)


def classify_question_type(question: str) -> str:
    """
    Returns "definitional" or "evidence_seeking" for a single (non-compound)
    question. Checks the definitional lead phrase first since it's the more
    specific signal; defaults to evidence_seeking when neither pattern is
    clearly present, since that's the system's original, primary behavior
    (safer default — under-restricting to background-only would silently
    drop real research questions that don't happen to use one of the
    evidence-seeking cue words).
    """
    text = (question or "").strip()
    if DEFINITIONAL_LEAD_PATTERN.search(text) and not EVIDENCE_SEEKING_CUE_PATTERN.search(text):
        return "definitional"
    return "evidence_seeking"


def decompose_compound(question: str) -> List[str]:
    """
    Splits a question into independent sub-questions if it genuinely asks
    more than one thing, e.g. "What is diabetes and does metformin help with
    weight loss?" -> ["What is diabetes", "does metformin help with weight
    loss?"]. Returns [question] unchanged (a list of one) when it isn't
    compound — including the specific case this was written to get right:
    "What is type 1 and type 2 diabetes" splits on " and " into "What is
    type 1" / "type 2 diabetes", but "type 2 diabetes" doesn't start with a
    question lead word, so it's recognized as a continuation of the same
    question (a compound *subject*, not a compound *question*) and the split
    is discarded.
    """
    text = (question or "").strip()
    if not text:
        return [text]

    parts = [p.strip() for p in _SPLIT_PATTERN.split(text) if p.strip()]
    if len(parts) < 2:
        return [text]

    # The first part is always kept (it's the start of the original
    # question, whether or not it itself "looks like" a new clause). Every
    # part after the first only counts as a separate sub-question if it
    # opens with its own interrogative/auxiliary lead word.
    sub_questions = [parts[0]]
    for part in parts[1:]:
        if _NEW_CLAUSE_LEAD_PATTERN.search(part):
            sub_questions.append(part)
        else:
            # Not a new question — reattach it to the previous sub-question
            # exactly as the conjunction joined it in the original text.
            sub_questions[-1] = f"{sub_questions[-1]} and {part}"

    return sub_questions if len(sub_questions) > 1 else [text]


def classify_and_decompose(question: str) -> List[dict]:
    """
    The single entry point src.rag.chain uses: decomposes `question` into
    sub-questions (a list of one if it isn't compound) and classifies each,
    returning [{"question": str, "type": "definitional"|"evidence_seeking"}, ...]
    in the order they should be answered — definitional sub-questions first,
    then evidence-seeking ones, per the "definition/background first, then
    what the studies show" rule. Original relative order is preserved within
    each type (stable sort).
    """
    sub_questions = decompose_compound(question)
    classified = [{"question": q, "type": classify_question_type(q)} for q in sub_questions]
    order = {"definitional": 0, "evidence_seeking": 1}
    classified.sort(key=lambda item: order[item["type"]])
    return classified


def overall_question_type(classified: List[dict]) -> str:
    """
    "definitional" / "evidence_seeking" if every sub-question agrees,
    else "compound" — the label surfaced in API payloads/logs.
    """
    types = {item["type"] for item in classified}
    if len(types) == 1:
        return types.pop()
    return "compound"
