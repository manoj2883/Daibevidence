from unittest.mock import patch

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from src.rag import hybrid_retriever, offline
from src.rag.hybrid_retriever import tokenize

# Four papers; paper "p1" has two chunks. Vectors are one-hot so dense ranking is controllable.
CHUNKS = [
    ("p1_0", "p1", "Mediterranean diet lowers type 2 diabetes risk in cohort studies."),
    ("p1_1", "p1", "Adherence to the Mediterranean diet was associated with lower incidence."),
    ("p2_0", "p2", "Metformin use and vitamin B12 deficiency in type 2 diabetes."),
    ("p3_0", "p3", "Bariatric surgery induces remission of type 2 diabetes."),
    ("bg_ada_prediabetes_0", "bg_ada_prediabetes", "Prediabetes means blood sugar is higher than normal."),
]
PARENTS = {
    "p1": {"source_type": "evidence", "pmid": "111", "title": "Med diet meta-analysis", "year": "2024",
           "study_design": "Meta-analysis", "population": "type2", "text": "Full abstract about the Mediterranean diet. " * 20},
    "p2": {"source_type": "evidence", "pmid": "222", "title": "Metformin and B12", "year": "2021",
           "study_design": "RCT", "population": "type2", "text": "B12 abstract."},
    "p3": {"source_type": "evidence", "pmid": "333", "title": "Surgery and remission", "year": "2025",
           "study_design": "Review", "population": "type2", "text": "Surgery abstract."},
    "bg_ada_prediabetes": {"source_type": "background", "publisher": "ADA", "title": "Prediabetes", "source_url": "https://diabetes.org",
                           "study_design": "Background (ADA)", "population": "prediabetes", "text": "Prediabetes page."},
}


class FakeEmbeddings:
    """Embeds any question to the vector of the first paper's chunks (dense favours p1)."""
    def embed_query(self, text):
        return [1.0, 0.0, 0.0, 0.0, 0.0]


@pytest.fixture
def fake_corpus(monkeypatch):
    chunks = [{"id": cid, "text": text, "metadata": {"parent_id": pid}} for cid, pid, text in CHUNKS]
    corpus = {"chunks": chunks, "parents": PARENTS, "bm25": BM25Okapi([tokenize(c["text"]) for c in chunks]),
              "by_id": {c["id"]: c for c in chunks}}
    monkeypatch.setattr(hybrid_retriever, "_corpus", corpus)
    vectors = np.eye(len(chunks), dtype=np.float32)
    vectors[1] = [0.9, 0.1, 0, 0, 0]  # p1's second chunk is also close to the query
    monkeypatch.setattr(offline, "_index", {"vectors": vectors, "ids": [c["id"] for c in chunks]})
    monkeypatch.setattr(offline, "LocalSentenceTransformerEmbeddings", FakeEmbeddings)
    return corpus


def test_offline_search_fuses_dense_and_bm25_one_entry_per_paper(fake_corpus, monkeypatch):
    monkeypatch.setattr(offline, "OFFLINE_CANDIDATES", 2)  # dense top 2 = p1's two chunks only
    results = offline.offline_search("bariatric surgery remission")
    ids = [r["pmid"] or r["publisher"] for r in results]
    assert len(ids) == len(set(ids))  # p1's two chunks collapse into one paper
    assert "111" in ids and "333" in ids  # dense finds p1, BM25 finds p3
    p3 = next(r for r in results if r["pmid"] == "333")
    assert p3["found_by"] == ["bm25"] and p3["evidence_badge"] == "Review" and p3["year"] == "2025"


def test_offline_results_carry_title_year_badge_and_snippet(fake_corpus):
    top = offline.offline_search("Mediterranean diet", k=1)[0]
    assert top["title"] == "Med diet meta-analysis" and top["year"] == "2024" and top["evidence_badge"] == "Meta-analysis"
    assert len(top["snippet"]) <= offline.SNIPPET_CHARS + 1 and top["snippet"].endswith("…")


def test_offline_mode_returns_five_studies_and_makes_no_api_calls(fake_corpus, monkeypatch):
    from src.rag import pipeline

    monkeypatch.setenv("OFFLINE_MODE", "1")
    boom = RuntimeError("network call in offline mode")
    with patch("anthropic.Anthropic", side_effect=boom), patch("pinecone.Pinecone", side_effect=boom), \
         patch.object(pipeline, "stream_answer_v3", side_effect=boom), patch.object(pipeline, "get_client", side_effect=boom):
        events = list(pipeline.run_pipeline("Does the Mediterranean diet lower diabetes risk?"))
    assert [e["event"] for e in events] == ["offline", "done"]
    data = events[0]["data"]
    assert data["message"] == "Offline mode: showing the most relevant studies"
    assert data["status"] == "offline" and 1 <= len(data["sources"]) <= 5


def test_offline_flag_parsing(monkeypatch):
    for value, expected in (("1", True), ("true", True), ("0", False), ("", False)):
        monkeypatch.setenv("OFFLINE_MODE", value)
        assert offline.offline_mode() is expected
