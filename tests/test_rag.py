import json
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


def test_refuses_when_nothing_clears_the_floor():
    """
    When nothing in the candidate pool clears the similarity floor, the
    chain must refuse without ever calling Claude, and surface the closest
    scores + scope description + score distribution.
    """
    from src.rag.types import RetrievalDecision

    decision = RetrievalDecision(
        state="refused",
        surviving_chunks=[],
        candidate_scores=[0.31, 0.28, 0.20],
        floor=0.5,
        low_confidence_margin=0.05,
    )

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-index"}):
        with patch("src.rag.chain.retrieve_with_floor", return_value=decision):
            with patch("src.rag.chain.get_client") as mock_get_client:
                from src.rag.chain import REFUSAL_TEXT, stream_answer

                events = list(stream_answer("What is the airspeed velocity of an unladen swallow?"))

                assert len(events) == 1
                assert events[0]["event"] == "refusal"
                data = events[0]["data"]
                assert data["message"] == REFUSAL_TEXT
                assert data["closest_scores"] == [0.31, 0.28, 0.20]
                assert "diet and nutrition" in data["scope_description"]
                assert data["score_distribution"]["floor"] == 0.5
                assert data["score_distribution"]["surviving_count"] == 0
                assert "medical advice" in data["disclaimer"].lower()
                mock_get_client.return_value.messages.stream.assert_not_called()


def test_generation_failure_yields_error_event_not_silent_death():
    """
    If Claude's API call itself fails (rate limit, usage cap, network blip),
    the stream must yield a clean "error" event after "sources" — not die
    as an unhandled exception that leaves the client hanging forever.
    """
    import anthropic
    import httpx2

    from src.rag.types import RetrievalDecision, RetrievedChunk

    chunk = RetrievedChunk(
        text="Some evidence text.",
        metadata={"pmid": "123", "title": "t", "journal": "j", "year": "2024", "publication_type": "Review", "population": "type2"},
        score=0.9,
    )
    decision = RetrievalDecision(
        state="answered",
        surviving_chunks=[chunk],
        candidate_scores=[0.9],
        floor=0.5,
        low_confidence_margin=0.05,
    )

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-index"}):
        with patch("src.rag.chain.retrieve_with_floor", return_value=decision):
            with patch("src.rag.chain.get_client") as mock_get_client:
                mock_client = mock_get_client.return_value
                fake_request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
                mock_client.messages.stream.side_effect = anthropic.APIConnectionError(request=fake_request)

                from src.rag.chain import stream_answer

                events = list(stream_answer("What does the evidence say about type 2 diabetes diet?"))

                assert [e["event"] for e in events] == ["sources", "error"]
                assert "Generation failed" in events[1]["data"]["message"]


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


def test_decide_retrieval_state_refuses_when_nothing_clears_floor():
    from src.rag.retriever import decide_retrieval_state
    from src.rag.types import RetrievedChunk

    candidates = [RetrievedChunk(text="x", score=s) for s in [0.4, 0.3, 0.2]]
    decision = decide_retrieval_state(candidates, floor=0.5, low_confidence_margin=0.05)

    assert decision.state == "refused"
    assert decision.surviving_chunks == []
    assert decision.candidate_scores == [0.4, 0.3, 0.2]
    assert decision.top_score is None


def test_decide_retrieval_state_low_confidence_near_floor():
    from src.rag.retriever import decide_retrieval_state
    from src.rag.types import RetrievedChunk

    candidates = [RetrievedChunk(text="x", score=s) for s in [0.52, 0.40]]
    decision = decide_retrieval_state(candidates, floor=0.5, low_confidence_margin=0.05)

    assert decision.state == "answered_low_confidence"
    assert len(decision.surviving_chunks) == 1
    assert decision.top_score == 0.52


def test_decide_retrieval_state_answered_well_above_floor():
    from src.rag.retriever import decide_retrieval_state
    from src.rag.types import RetrievedChunk

    candidates = [RetrievedChunk(text="x", score=s) for s in [0.72, 0.65, 0.30]]
    decision = decide_retrieval_state(candidates, floor=0.5, low_confidence_margin=0.05)

    assert decision.state == "answered"
    assert len(decision.surviving_chunks) == 2
    assert decision.top_score == 0.72


def test_extract_contradiction_block_not_yet_complete():
    from src.rag.chain import extract_contradiction_block

    assert extract_contradiction_block("<<<CONTRADICTIONS>>>\n[") is None
    assert extract_contradiction_block("") is None


def test_extract_contradiction_block_empty_array():
    from src.rag.chain import extract_contradiction_block

    buffer = "<<<CONTRADICTIONS>>>\n[]\n<<<END_CONTRADICTIONS>>>\n\nThe answer starts here."
    result = extract_contradiction_block(buffer)
    assert result is not None
    contradictions, rest = result
    assert contradictions == []
    assert rest == "\n\nThe answer starts here."


def test_extract_contradiction_block_with_entries():
    from src.rag.chain import extract_contradiction_block

    entries = [{"excerpt_indices": [1, 2], "claim": "c", "position_a": "a", "position_b": "b", "differs_by": "duration"}]
    buffer = "<<<CONTRADICTIONS>>>\n" + json.dumps(entries) + "\n<<<END_CONTRADICTIONS>>>\nAnswer text."
    contradictions, rest = extract_contradiction_block(buffer)
    assert contradictions == entries
    assert rest == "\nAnswer text."


def test_extract_contradiction_block_malformed_json_degrades_gracefully():
    from src.rag.chain import extract_contradiction_block

    buffer = "<<<CONTRADICTIONS>>>\nnot valid json at all\n<<<END_CONTRADICTIONS>>>\nAnswer."
    contradictions, rest = extract_contradiction_block(buffer)
    assert contradictions == []
    assert rest == "\nAnswer."


def test_extract_contradiction_block_missing_start_marker():
    from src.rag.chain import extract_contradiction_block

    buffer = "some stray text<<<END_CONTRADICTIONS>>>\nAnswer."
    contradictions, rest = extract_contradiction_block(buffer)
    assert contradictions == []
    assert rest == "\nAnswer."


def test_resolve_contradiction_pmids_maps_indices_and_drops_invalid():
    from src.rag.chain import _resolve_contradiction_pmids
    from src.rag.types import RetrievedChunk

    chunks = [
        RetrievedChunk(text="a", metadata={"pmid": "111"}, score=0.9),
        RetrievedChunk(text="b", metadata={"pmid": "222"}, score=0.8),
    ]
    entries = [
        {"excerpt_indices": [1, 2], "claim": "c", "position_a": "a", "position_b": "b", "differs_by": "d"},
        {"excerpt_indices": [1, 99], "claim": "bad index"},  # out of range — dropped
        {"excerpt_indices": "not-a-list", "claim": "bad type"},  # dropped
        "not-a-dict",  # dropped
    ]
    resolved = _resolve_contradiction_pmids(entries, chunks)
    assert len(resolved) == 1
    assert resolved[0]["pmids"] == ["111", "222"]
    assert resolved[0]["claim"] == "c"


def test_stream_answer_emits_contradictions_before_tokens():
    """
    Full stream_answer flow with a mocked Claude stream that emits the
    contradiction preamble first: contradictions must be parsed off and
    sent as their own event before any prose reaches "token" events, and
    the parsed contradiction must carry the resolved PMIDs.
    """
    from unittest.mock import MagicMock

    from src.rag.types import RetrievalDecision, RetrievedChunk

    chunk1 = RetrievedChunk(
        text="Diet A improved HbA1c.",
        metadata={"pmid": "111", "title": "t1", "journal": "j", "year": "2024", "publication_type": "RCT", "population": "type2"},
        score=0.9,
    )
    chunk2 = RetrievedChunk(
        text="Diet A had no effect on HbA1c.",
        metadata={"pmid": "222", "title": "t2", "journal": "j", "year": "2023", "publication_type": "RCT", "population": "type2"},
        score=0.85,
    )
    decision = RetrievalDecision(
        state="answered",
        surviving_chunks=[chunk1, chunk2],
        candidate_scores=[0.9, 0.85],
        floor=0.5,
        low_confidence_margin=0.05,
    )

    contradiction_entry = {
        "excerpt_indices": [1, 2],
        "claim": "effect of Diet A on HbA1c",
        "position_a": "improved HbA1c",
        "position_b": "no effect on HbA1c",
        "differs_by": "study duration",
    }
    stream_chunks = [
        "<<<CONTRADICTIONS>>>\n",
        json.dumps([contradiction_entry]),
        "\n<<<END_CONTRADICTIONS>>>\n\nThe evidence disagrees ",
        "on this point [PMID: 111][PMID: 222].",
    ]

    mock_stream_cm = MagicMock()
    mock_stream_cm.__enter__.return_value.text_stream = iter(stream_chunks)
    mock_stream_cm.__exit__.return_value = False

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-index"}):
        with patch("src.rag.chain.retrieve_with_floor", return_value=decision):
            with patch("src.rag.chain.get_client") as mock_get_client:
                mock_get_client.return_value.messages.stream.return_value = mock_stream_cm

                from src.rag.chain import stream_answer

                events = list(stream_answer("Does Diet A improve HbA1c in type 2 diabetes?"))

    event_types = [e["event"] for e in events]
    assert event_types[0] == "sources"
    assert event_types[1] == "contradictions"
    assert "token" in event_types
    assert event_types[-1] == "done"

    contradictions = events[1]["data"]["contradictions"]
    assert len(contradictions) == 1
    assert contradictions[0]["pmids"] == ["111", "222"]
    assert contradictions[0]["claim"] == "effect of Diet A on HbA1c"

    token_events = [e for e in events if e["event"] == "token"]
    full_text = "".join(e["data"]["text"] for e in token_events)
    assert "CONTRADICTIONS" not in full_text
    assert "The evidence disagrees on this point" in full_text


def test_stream_answer_emits_empty_contradictions_when_none_found():
    from unittest.mock import MagicMock

    from src.rag.types import RetrievalDecision, RetrievedChunk

    chunk = RetrievedChunk(
        text="Diet A improved HbA1c.",
        metadata={"pmid": "111", "title": "t1", "journal": "j", "year": "2024", "publication_type": "RCT", "population": "type2"},
        score=0.9,
    )
    decision = RetrievalDecision(
        state="answered", surviving_chunks=[chunk], candidate_scores=[0.9], floor=0.5, low_confidence_margin=0.05,
    )

    stream_chunks = ["<<<CONTRADICTIONS>>>\n[]\n<<<END_CONTRADICTIONS>>>\n\nDiet A improved HbA1c [PMID: 111]."]

    mock_stream_cm = MagicMock()
    mock_stream_cm.__enter__.return_value.text_stream = iter(stream_chunks)
    mock_stream_cm.__exit__.return_value = False

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-index"}):
        with patch("src.rag.chain.retrieve_with_floor", return_value=decision):
            with patch("src.rag.chain.get_client") as mock_get_client:
                mock_get_client.return_value.messages.stream.return_value = mock_stream_cm

                from src.rag.chain import stream_answer

                events = list(stream_answer("Does Diet A improve HbA1c?"))

    assert events[1]["event"] == "contradictions"
    assert events[1]["data"]["contradictions"] == []


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
