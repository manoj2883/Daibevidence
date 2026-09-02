import os
from unittest.mock import patch

import pytest

from src.ingest.chunker import split_pubmed_documents, split_text_by_words


def test_split_text_by_words():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = split_text_by_words(text, chunk_size_words=300, overlap_words=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 300
    # Consecutive chunks overlap by the requested number of words.
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert first_words[-50:] == second_words[:50]

def test_split_text_by_words_empty():
    assert split_text_by_words("") == []

def test_split_pubmed_documents():
    documents = [
        {
            "pmid": "12345",
            "title": "Diet and glycemic control in type 2 diabetes",
            "journal": "Diabetes Care",
            "year": "2023",
            "publication_type": "Systematic Review",
            "population": "type2",
            "abstract": " ".join(f"finding{i}" for i in range(650)),
        }
    ]
    chunks = split_pubmed_documents(documents, chunk_size_words=300, overlap_words=50)
    assert len(chunks) >= 2
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        assert meta["pmid"] == "12345"
        assert meta["title"] == "Diet and glycemic control in type 2 diabetes"
        assert meta["journal"] == "Diabetes Care"
        assert meta["year"] == "2023"
        assert meta["publication_type"] == "Systematic Review"
        assert meta["population"] == "type2"
        assert meta["chunk_index"] == i


def test_refuses_when_no_evidence_retrieved():
    """
    When retrieval returns nothing, the chain must refuse without ever calling Claude.
    """
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-index"}):
        with patch("src.rag.chain.retrieve", return_value=[]):
            with patch("src.rag.chain.get_client") as mock_get_client:
                from src.rag.chain import REFUSAL_TEXT, stream_answer

                events = list(stream_answer("What is the airspeed velocity of an unladen swallow?"))

                assert len(events) == 1
                assert events[0]["event"] == "refusal"
                assert events[0]["data"]["message"] == REFUSAL_TEXT
                assert "medical advice" in events[0]["data"]["disclaimer"].lower()
                mock_get_client.return_value.messages.stream.assert_not_called()


def test_format_context_and_retrieved_populations():
    from src.rag.chain import format_context, retrieved_populations_summary
    from src.rag.types import RetrievedChunk

    chunks = [
        RetrievedChunk(
            text="Low-carbohydrate diets improved HbA1c by 0.5%.",
            metadata={
                "pmid": "11111111",
                "title": "Diet and glycemic control",
                "journal": "Diabetes Care",
                "year": "2022",
                "publication_type": "Clinical Trial",
                "population": "type2",
            },
            score=0.91,
        ),
        RetrievedChunk(
            text="Weight loss interventions in pregnancy.",
            metadata={
                "pmid": "22222222",
                "title": "GDM outcomes",
                "journal": "Obstet Gynecol",
                "year": "2021",
                "publication_type": "Systematic Review",
                "population": "gestational",
            },
            score=0.85,
        ),
    ]

    context = format_context(chunks)
    assert "PMID 11111111" in context
    assert "PMID 22222222" in context
    assert "Population: type2" in context
    assert "Population: gestational" in context

    assert retrieved_populations_summary(chunks) == "gestational, type2"


def test_population_mismatch_flag():
    from src.rag.chain import is_population_mismatch

    assert is_population_mismatch("gestational", {"type2"}) is True
    assert is_population_mismatch("type2", {"type2", "mixed"}) is False
    assert is_population_mismatch("mixed", {"type2"}) is False
    assert is_population_mismatch("type2", set()) is False


def test_assert_chunk_cap_raises_over_limit():
    from src.rag.cost import assert_chunk_cap
    from src.rag.types import RetrievedChunk

    chunks = [RetrievedChunk(text="x") for _ in range(6)]
    with pytest.raises(ValueError):
        assert_chunk_cap(chunks, max_chunks=5)


def test_assert_chunk_cap_allows_at_limit():
    from src.rag.cost import assert_chunk_cap
    from src.rag.types import RetrievedChunk

    chunks = [RetrievedChunk(text="x") for _ in range(5)]
    assert_chunk_cap(chunks, max_chunks=5)  # should not raise


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="Anthropic API key not configured in environment"
)
def test_grounded_answer_generation_live():
    """
    End-to-end smoke test against the real Claude + Pinecone APIs. Only runs
    when ANTHROPIC_API_KEY (and Pinecone config) are present in the environment.
    """
    from src.rag.chain import stream_answer

    events = list(stream_answer("What does the evidence say about diet and glycemic control in type 2 diabetes?"))
    event_types = [e["event"] for e in events]

    assert "sources" in event_types
    assert "token" in event_types
    assert event_types[-1] == "done"
