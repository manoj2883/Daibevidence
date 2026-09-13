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
        state="out_of_scope",
        surviving_chunks=[],
        candidate_scores=[0.31, 0.28, 0.20],
        floor=0.5,
        low_confidence_margin=0.05,
    )

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-index"}):
        with patch("src.rag.chain.retrieve_with_floor", return_value=decision):
            with patch("src.rag.chain.get_client") as mock_get_client:
                from src.rag.chain import OUT_OF_SCOPE_TEXT, stream_answer

                events = list(stream_answer("What is the airspeed velocity of an unladen swallow?"))

                assert len(events) == 1
                assert events[0]["event"] == "refusal"
                data = events[0]["data"]
                assert data["state"] == "out_of_scope"
                assert data["message"] == OUT_OF_SCOPE_TEXT
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
    assert "SOURCE TYPE: evidence" in context
    assert "PMID 11111111" in context
    assert "PMID 22222222" in context
    assert "Population: type2" in context
    assert "Population: gestational" in context

    assert retrieved_populations_summary(chunks) == "gestational, type2"


def test_format_context_labels_background_chunks_distinctly():
    from src.rag.chain import format_context
    from src.rag.types import RetrievedChunk

    chunks = [
        RetrievedChunk(
            text="Type 2 diabetes is when the body doesn't use insulin properly.",
            metadata={
                "source_type": "background",
                "publisher": "ADA",
                "title": "Type 2 Diabetes",
                "source_url": "https://diabetes.org/about-diabetes/type-2",
                "population": "type2",
            },
            score=0.7,
        ),
    ]
    context = format_context(chunks)
    assert "SOURCE TYPE: background" in context
    assert "Publisher: ADA" in context
    assert "PMID" not in context  # background excerpts have no PMID at all


def test_source_payload_includes_source_type_and_publisher():
    from src.rag.chain import _source_payload
    from src.rag.types import RetrievedChunk

    evidence_chunk = RetrievedChunk(text="x", metadata={"pmid": "1"}, score=0.9)
    assert _source_payload(evidence_chunk)["source_type"] == "evidence"

    background_chunk = RetrievedChunk(
        text="x", metadata={"source_type": "background", "publisher": "CDC", "source_url": "https://cdc.gov/x"}, score=0.6,
    )
    payload = _source_payload(background_chunk)
    assert payload["source_type"] == "background"
    assert payload["publisher"] == "CDC"
    assert payload["source_url"] == "https://cdc.gov/x"


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

    assert decision.state == "out_of_scope"
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


def test_find_complete_json_objects_incremental():
    from src.rag.chain import find_complete_json_objects

    # Nothing complete yet — scan stops at the start of the incomplete
    # object (position 1, past the leading "["), ready to re-scan from
    # there once more text streams in.
    objs, pos = find_complete_json_objects('[{"sentence": "a"', 0)
    assert objs == []
    assert pos == 1

    # One complete object, resume position lands right after it.
    buf = '[{"sentence": "a"}'
    objs, pos = find_complete_json_objects(buf, 0)
    assert objs == ['{"sentence": "a"}']
    assert pos == len(buf)

    # Braces inside a quoted string must not confuse depth tracking.
    buf2 = '{"sentence": "a { fake brace } and \\"quote\\""}'
    objs2, pos2 = find_complete_json_objects(buf2, 0)
    assert objs2 == [buf2]


def test_find_complete_json_objects_skips_leading_stray_text():
    """
    Regression test: a long contradiction analysis that never closes before
    MAX_PREAMBLE_BUFFER triggers dumps leftover "<<<CONTRADICTIONS>>>"
    marker text (or any other stray prose) at the front of the sentence
    buffer. The scanner must skip past it and still find real sentence
    objects later in the buffer, rather than permanently giving up on the
    first unexpected character (that was a real bug: it produced zero
    sentences from a real generation that actually contained content).
    """
    from src.rag.chain import find_complete_json_objects

    buffer = '<<<CONTRADICTIONS>>>\n[{"excerpt_indices": [1]}]garbage text{"sentence": "Real content.", "chunk_ids": [1], "supported": true}'
    objs, pos = find_complete_json_objects(buffer, 0)

    # Both the (wrongly-shaped) contradiction-like object and the real
    # sentence object get returned as raw text — _validate_sentence()
    # filters the former out downstream; the scanner's job is just to not
    # get stuck.
    assert len(objs) == 2
    assert '"sentence": "Real content."' in objs[1]
    assert pos == len(buffer)

    # Resuming from a prior position only finds the new object.
    buf3 = '[{"sentence": "a"},{"sentence": "b"}]'
    first_objs, mid_pos = find_complete_json_objects(buf3, 0)
    assert len(first_objs) == 2
    more_objs, _ = find_complete_json_objects(buf3, mid_pos)
    assert more_objs == []


def test_validate_sentence_coerces_and_rejects():
    from src.rag.chain import _validate_sentence

    assert _validate_sentence({"sentence": "  x  ", "chunk_ids": [1, 2], "supported": True}) == {
        "sentence": "x", "chunk_ids": [1, 2], "supported": True,
    }
    # Missing supported -> derived from chunk_ids presence.
    assert _validate_sentence({"sentence": "x", "chunk_ids": [1]})["supported"] is True
    assert _validate_sentence({"sentence": "x", "chunk_ids": []})["supported"] is False
    # Non-list chunk_ids -> coerced to [].
    assert _validate_sentence({"sentence": "x", "chunk_ids": "bad"})["chunk_ids"] == []
    # No sentence text at all -> rejected.
    assert _validate_sentence({"chunk_ids": [1]}) is None
    assert _validate_sentence({"sentence": "   "}) is None
    assert _validate_sentence("not a dict") is None


def test_compute_groundedness():
    from src.rag.chain import compute_groundedness

    sentences = [
        {"sentence": "a", "chunk_ids": [1], "supported": True},
        {"sentence": "b", "chunk_ids": [], "supported": False},
        {"sentence": "c", "chunk_ids": [2, 3], "supported": True},
    ]
    result = compute_groundedness(sentences)
    assert result["supported_sentences"] == 2
    assert result["total_sentences"] == 3
    assert result["groundedness"] == round(2 / 3, 4)

    assert compute_groundedness([])["groundedness"] is None


def _make_decision(chunks, floor=0.5, margin=0.05):
    from src.rag.types import RetrievalDecision
    return RetrievalDecision(
        state="answered", surviving_chunks=chunks,
        candidate_scores=[c.score for c in chunks], floor=floor, low_confidence_margin=margin,
    )


def test_determine_final_state_no_evidence_backed_sentence_is_no_evidence_for_claim():
    from src.rag.chain import determine_final_state
    from src.rag.types import RetrievedChunk

    evidence_chunk = RetrievedChunk(text="e", metadata={"source_type": "evidence"}, score=0.9)
    background_chunk = RetrievedChunk(text="b", metadata={"source_type": "background"}, score=0.8)
    chunks = [evidence_chunk, background_chunk]
    decision = _make_decision(chunks)

    # Only the background chunk (index 2) actually got cited — the model
    # correctly declined to use the evidence chunk (index 1) even though it
    # cleared the floor.
    sentences = [
        {"sentence": "Definition here.", "chunk_ids": [2], "supported": True},
        {"sentence": "No study addresses the specific claim.", "chunk_ids": [], "supported": False},
    ]
    assert determine_final_state(chunks, sentences, decision) == "no_evidence_for_claim"


def test_determine_final_state_evidence_backed_is_answered():
    from src.rag.chain import determine_final_state
    from src.rag.types import RetrievedChunk

    evidence_chunk = RetrievedChunk(text="e", metadata={"source_type": "evidence"}, score=0.9)
    background_chunk = RetrievedChunk(text="b", metadata={"source_type": "background"}, score=0.8)
    chunks = [evidence_chunk, background_chunk]
    decision = _make_decision(chunks)

    sentences = [
        {"sentence": "Definitional framing.", "chunk_ids": [2], "supported": True},
        {"sentence": "The trial found an effect.", "chunk_ids": [1], "supported": True},
    ]
    assert determine_final_state(chunks, sentences, decision) == "answered"


def test_determine_final_state_evidence_backed_but_weak_score_is_low_confidence():
    from src.rag.chain import determine_final_state
    from src.rag.types import RetrievedChunk

    evidence_chunk = RetrievedChunk(text="e", metadata={"source_type": "evidence"}, score=0.51)
    chunks = [evidence_chunk]
    decision = _make_decision(chunks, floor=0.5, margin=0.05)

    sentences = [{"sentence": "Weak evidence claim.", "chunk_ids": [1], "supported": True}]
    assert determine_final_state(chunks, sentences, decision) == "answered_low_confidence"


def test_provisional_state_background_only_survivors_is_no_evidence_for_claim():
    from src.rag.chain import _provisional_state
    from src.rag.types import RetrievedChunk

    background_chunk = RetrievedChunk(text="b", metadata={"source_type": "background"}, score=0.8)
    decision = _make_decision([background_chunk])
    assert _provisional_state(decision) == "no_evidence_for_claim"


def test_provisional_state_evidence_survivor_is_answered():
    from src.rag.chain import _provisional_state
    from src.rag.types import RetrievedChunk

    evidence_chunk = RetrievedChunk(text="e", metadata={"source_type": "evidence"}, score=0.9)
    decision = _make_decision([evidence_chunk])
    assert _provisional_state(decision) == "answered"


def test_stream_answer_emits_contradictions_then_sentences():
    """
    Full stream_answer flow with a mocked Claude stream that emits the
    contradiction preamble, then a JSON array of sentences: contradictions
    must be parsed off and sent as their own event before any "sentence"
    events, each sentence must carry resolved PMIDs, and "done" must carry
    the groundedness summary.
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
    sentence_1 = {"sentence": "One study found Diet A improved HbA1c.", "chunk_ids": [1], "supported": True}
    sentence_2 = {"sentence": "Another found no effect.", "chunk_ids": [2], "supported": True}
    sentence_3 = {"sentence": "In short, the evidence disagrees.", "chunk_ids": [], "supported": False}

    stream_chunks = [
        "<<<CONTRADICTIONS>>>\n",
        json.dumps([contradiction_entry]),
        "\n<<<END_CONTRADICTIONS>>>\n[",
        json.dumps(sentence_1) + ",",
        json.dumps(sentence_2) + ",",
        json.dumps(sentence_3) + "]",
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
    assert event_types.count("sentence") == 3
    assert event_types[-1] == "done"

    contradictions = events[1]["data"]["contradictions"]
    assert len(contradictions) == 1
    assert contradictions[0]["pmids"] == ["111", "222"]

    sentence_events = [e["data"] for e in events if e["event"] == "sentence"]
    assert sentence_events[0]["pmids"] == ["111"]
    assert sentence_events[1]["pmids"] == ["222"]
    assert sentence_events[2]["supported"] is False
    assert sentence_events[2]["pmids"] == []

    done_data = events[-1]["data"]
    assert done_data["groundedness"]["total_sentences"] == 3
    assert done_data["groundedness"]["supported_sentences"] == 2


def test_stream_answer_plain_text_noncompliance_still_degrades_gracefully():
    """
    The model is always expected to use the JSON array format now (the old
    "reply with exactly this sentence, no JSON at all" refusal path was
    retired along with the two-state refusal). If it ever ignores that
    instruction and emits plain text anyway, the stream must still degrade
    into the documented {sentence, chunk_ids, supported} shape rather than
    losing the response — and because no evidence chunk ends up cited, the
    authoritative final state must come out as no_evidence_for_claim.
    """
    from unittest.mock import MagicMock

    from src.rag.types import RetrievalDecision, RetrievedChunk

    plain_text_response = "Sorry, I can't answer that from the given excerpts."

    chunk = RetrievedChunk(
        text="Some unrelated excerpt.",
        metadata={"pmid": "111", "title": "t1", "journal": "j", "year": "2024", "publication_type": "RCT", "population": "type2"},
        score=0.9,
    )
    decision = RetrievalDecision(
        state="answered", surviving_chunks=[chunk], candidate_scores=[0.9], floor=0.5, low_confidence_margin=0.05,
    )

    mock_stream_cm = MagicMock()
    mock_stream_cm.__enter__.return_value.text_stream = iter([plain_text_response])
    mock_stream_cm.__exit__.return_value = False

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-index"}):
        with patch("src.rag.chain.retrieve_with_floor", return_value=decision):
            with patch("src.rag.chain.get_client") as mock_get_client:
                mock_get_client.return_value.messages.stream.return_value = mock_stream_cm

                from src.rag.chain import stream_answer

                events = list(stream_answer("An unanswerable question."))

    event_types = [e["event"] for e in events]
    assert event_types == ["sources", "contradictions", "sentence", "done"]

    sentence_data = [e for e in events if e["event"] == "sentence"][0]["data"]
    assert sentence_data["sentence"] == plain_text_response
    assert sentence_data["chunk_ids"] == []
    assert sentence_data["supported"] is False

    done_data = [e for e in events if e["event"] == "done"][0]["data"]
    assert done_data["state"] == "no_evidence_for_claim"


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
