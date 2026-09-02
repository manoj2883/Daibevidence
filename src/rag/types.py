"""
Plain data types shared across retrieval and generation. Replaces
LangChain's Document — no framework dependency, just what the pipeline
actually needs: a chunk's text, its metadata, and its similarity score.
"""
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class RetrievedChunk:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
