"""
Plain data types shared across retrieval and generation. Replaces
LangChain's Document — no framework dependency, just what the pipeline
actually needs: a chunk's text, its metadata, and its similarity score.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RetrievedChunk:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class RetrievalDecision:
    """
    Result of applying the similarity floor to a candidate pool. `state` is
    one of "answered", "answered_low_confidence", or "out_of_scope" —
    computed from scores alone, before any model call.
    """
    state: str
    surviving_chunks: List[RetrievedChunk]
    candidate_scores: List[float]  # every candidate's score, descending
    floor: float  # the evidence-tier floor
    low_confidence_margin: float
    background_floor: Optional[float] = None  # Phase 4: tuned independently; None means "same as floor"

    @property
    def top_score(self) -> Optional[float]:
        return self.surviving_chunks[0].score if self.surviving_chunks else None
