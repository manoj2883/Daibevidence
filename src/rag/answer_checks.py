"""
Deterministic checks on a generated answer, run after generation and before anything is shown.
No model calls.

In order:
1. A citation that doesn't match a passage retrieved for this question (a chunk_id outside the
   passage list, or a "PMID n" in the text that isn't one of the retrieved papers) -> not_covered.
   This should never happen, so it is logged as a bug.
2. A factual sentence with no citation -> the sentence is removed and the answer becomes
   answered_partial. Sentences about the retrieved evidence itself are not factual claims and are
   kept: what the studies don't show, which population they cover, or that a comparison spans
   separate studies (EVIDENCE_META below; written from the 24 uncited sentences in the v3 Run B
   answers, 23 of which were of this kind).
3. Zero citations left in the answer -> not_covered.
"""
import re
from typing import Any, Dict, Iterable, List

EVIDENCE_META = re.compile(
    r"\b(?:studies|study|evidence|research|findings|information|topic|comparison|sources)\s+"
    r"(?:gathered|found here|reviewed here|described (?:here|below|above)|retrieved|here|covers?|"
    r"comes? from|applies to|apply to|has been studied|have been studied|is pieced|are pieced|was pieced)\b"
    r"|\bseparate studies\b|\bno (?:single )?stud(?:y|ies)\b|\bnone of the (?:studies|retrieved)\b",
    re.IGNORECASE,
)
PMID_IN_TEXT = re.compile(r"\bPMID[:\s#]*(\d{5,9})\b", re.IGNORECASE)


def is_evidence_meta(sentence: str) -> bool:
    return bool(EVIDENCE_META.search(sentence or ""))


def check_answer(sentences: List[Dict[str, Any]], n_passages: int, retrieved_pmids: Iterable[str]) -> Dict[str, Any]:
    """
    Returns {"outcome": "ok" | "partial" | "not_covered", "keep": [indices of sentences to show],
    "removed": [sentence texts], "bad_citations": [...], "reason": str}.
    """
    retrieved = {str(p) for p in retrieved_pmids if p}
    bad = []
    for i, s in enumerate(sentences):
        for cid in s.get("chunk_ids") or []:
            if not isinstance(cid, int) or not 1 <= cid <= n_passages:
                bad.append({"sentence_index": i, "chunk_id": cid})
        for pmid in PMID_IN_TEXT.findall(s.get("sentence") or ""):
            if pmid not in retrieved:
                bad.append({"sentence_index": i, "pmid": pmid})
    if bad:
        return {"outcome": "not_covered", "keep": [], "removed": [], "bad_citations": bad,
                "reason": "citation does not match a retrieved passage"}

    keep, removed = [], []
    for i, s in enumerate(sentences):
        if s.get("chunk_ids") or is_evidence_meta(s.get("sentence")):
            keep.append(i)
        else:
            removed.append(s.get("sentence", ""))

    if not any(sentences[i].get("chunk_ids") for i in keep):
        return {"outcome": "not_covered", "keep": [], "removed": removed, "bad_citations": [],
                "reason": "no sentence cites a retrieved passage"}
    if removed:
        return {"outcome": "partial", "keep": keep, "removed": removed, "bad_citations": [],
                "reason": f"removed {len(removed)} uncited factual sentence(s)"}
    return {"outcome": "ok", "keep": keep, "removed": [], "bad_citations": [], "reason": ""}
