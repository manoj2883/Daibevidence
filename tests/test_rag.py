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


_IN_CHARTER = {"scope": "in_charter", "topic": "diet_nutrition", "reason": "ok", "model": "m"}
_ADJACENT = {"scope": "adjacent", "topic": None, "reason": "diabetes-adjacent but outside the four areas", "model": "m"}
_UNRELATED = {"scope": "unrelated", "topic": None, "reason": "not about diabetes", "model": "m"}


def _evidence(verdict, reason="ok", direct_map=None, population_mismatch=False, question_population=None, evidence_populations=None):
    """direct_map: {proposition_claim: [chunk_indices]} — builds a minimal propositions list."""
    propositions = []
    for claim, direct in (direct_map or {}).items():
        propositions.append({"claim": claim, "direct": direct, "partial": [], "absent": not direct})
    return {
        "propositions": propositions,
        "question_population": question_population,
        "evidence_populations": evidence_populations or [],
        "population_mismatch": population_mismatch,
        "verdict": verdict,
        "reason": reason,
        "model": "m",
    }


def test_refuses_when_scope_is_unrelated():
    """
    Stage A alone can refuse — no retrieval, no Stage B — when the
    question has nothing to do with diabetes.
    """
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-index"}):
        with patch("src.rag.chain.judge_scope", return_value=_UNRELATED):
            with patch("src.rag.chain._retrieve_for_question") as mock_retrieve:
                with patch("src.rag.chain.judge_evidence") as mock_evidence:
                    with patch("src.rag.chain.get_client") as mock_get_client:
                        from src.rag.chain import OUT_OF_SCOPE_TEXT, stream_answer

                        events = list(stream_answer("What is the capital of Australia?"))

    assert len(events) == 1
    assert events[0]["event"] == "refusal"
    data = events[0]["data"]
    assert data["status"] == "out_of_scope"
    assert data["state"] == "out_of_scope"
    assert data["message"] == OUT_OF_SCOPE_TEXT
    assert data["scope"] == "unrelated"
    assert data["evidence_verdict"] is None
    assert data["scope_model"] == "m"  # from _UNRELATED's "model" field — never dropped for the refusal path
    assert "medical advice" in data["disclaimer"].lower()
    mock_retrieve.assert_not_called()
    mock_evidence.assert_not_called()
    mock_get_client.return_value.messages.stream.assert_not_called()


def test_refuses_when_nothing_was_retrieved_at_all():
    """
    Defensive fallback: when retrieval returns literally nothing (e.g. an
    empty namespace) for an in_charter/adjacent question, the chain must
    refuse without ever calling Claude for generation — including never
    calling Stage B, since there's nothing to audit.
    """
    from src.rag.types import RetrievalDecision

    decision = RetrievalDecision(
        state="out_of_scope",
        surviving_chunks=[],
        candidate_scores=[0.31, 0.28, 0.20],
        floor=0.5,
        low_confidence_margin=0.05,
    )
    classified = [{"question": "What is the airspeed velocity of an unladen swallow?", "type": "evidence_seeking"}]

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-index"}):
        with patch("src.rag.chain.judge_scope", return_value=_IN_CHARTER):
            with patch("src.rag.chain._retrieve_for_question", return_value=(decision, classified)):
                with patch("src.rag.chain.judge_evidence") as mock_evidence:
                    with patch("src.rag.chain.get_client") as mock_get_client:
                        from src.rag.chain import NOTHING_RETRIEVED_TEXT, stream_answer

                        events = list(stream_answer("What is the airspeed velocity of an unladen swallow?"))

                        assert len(events) == 1
                        assert events[0]["event"] == "refusal"
                        data = events[0]["data"]
                        assert data["status"] == "out_of_scope"
                        assert data["state"] == "out_of_scope"
                        assert data["message"] == NOTHING_RETRIEVED_TEXT
                        assert data["closest_scores"] == [0.31, 0.28, 0.20]
                        assert "diet and nutrition" in data["scope_description"]
                        assert data["score_distribution"]["floor"] == 0.5
                        assert data["score_distribution"]["surviving_count"] == 0
                        assert data["evidence_verdict"] is None
                        assert "medical advice" in data["disclaimer"].lower()
                        mock_evidence.assert_not_called()
                        mock_get_client.return_value.messages.stream.assert_not_called()


def test_in_scope_no_evidence_when_in_charter_and_insufficient():
    """
    The status this rebuild exists to add: an in-charter question with
    thin retrieval must be told the corpus lacks evidence — never labeled
    out_of_scope, since the question itself is exactly what this system is
    meant to answer.
    """
    from src.rag.types import RetrievalDecision, RetrievedChunk

    chunk = RetrievedChunk(text="Unrelated tangential excerpt.", metadata={"pmid": "1", "population": "type2"}, score=0.6)
    decision = RetrievalDecision(state="answered", surviving_chunks=[chunk], candidate_scores=[0.6], floor=0.5, low_confidence_margin=0.05)
    classified = [{"question": "q", "type": "evidence_seeking"}]
    evidence_result = _evidence("insufficient", reason="No excerpt states the core proposition.")

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-index"}):
        with patch("src.rag.chain.judge_scope", return_value=_IN_CHARTER):
            with patch("src.rag.chain._retrieve_for_question", return_value=(decision, classified)):
                with patch("src.rag.chain.judge_evidence", return_value=evidence_result):
                    with patch("src.rag.chain.get_client") as mock_get_client:
                        from src.rag.chain import IN_SCOPE_NO_EVIDENCE_TEXT, stream_answer

                        events = list(stream_answer("q"))

    assert len(events) == 1
    data = events[0]["data"]
    assert data["status"] == "in_scope_no_evidence"
    assert data["message"] == IN_SCOPE_NO_EVIDENCE_TEXT
    assert "out of scope" not in data["message"].lower()
    assert data["evidence_verdict"] == "insufficient"
    mock_get_client.return_value.messages.stream.assert_not_called()


def test_out_of_scope_when_adjacent_and_insufficient():
    """adjacent + insufficient routes to out_of_scope, not in_scope_no_evidence."""
    from src.rag.types import RetrievalDecision, RetrievedChunk

    chunk = RetrievedChunk(text="Retinopathy laser treatment reduces vision loss risk.", metadata={"pmid": "999", "population": "type2"}, score=0.72)
    decision = RetrievalDecision(state="answered", surviving_chunks=[chunk], candidate_scores=[0.72], floor=0.5, low_confidence_margin=0.05)
    classified = [{"question": "How does diet affect glycemic control?", "type": "evidence_seeking"}]
    evidence_result = _evidence("insufficient", reason="The excerpt is about retinopathy treatment, not diet or glycemic control.")

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-index"}):
        with patch("src.rag.chain.judge_scope", return_value=_ADJACENT):
            with patch("src.rag.chain._retrieve_for_question", return_value=(decision, classified)):
                with patch("src.rag.chain.judge_evidence", return_value=evidence_result):
                    with patch("src.rag.chain.get_client") as mock_get_client:
                        from src.rag.chain import OUT_OF_SCOPE_TEXT, stream_answer

                        events = list(stream_answer("How does diet affect glycemic control?"))

    assert len(events) == 1
    data = events[0]["data"]
    assert data["status"] == "out_of_scope"
    assert data["message"] == OUT_OF_SCOPE_TEXT
    assert data["scope"] == "adjacent"
    assert data["evidence_verdict"] == "insufficient"
    mock_get_client.return_value.messages.stream.assert_not_called()


def test_retrieve_for_question_floor_mode_selects_only_floor_survivors():
    """
    retrieval_mode="floor" is diagnostic-only (used by
    scripts/run_old_pipeline_full.py to reconstruct the pre-judge
    pipeline) — it must select decide_retrieval_state's surviving_chunks,
    not the top-k-by-raw-score the default "topk" mode uses.
    """
    from src.rag.chain import _retrieve_for_question
    from src.rag.types import RetrievalDecision, RetrievedChunk

    c1 = RetrievedChunk(text="a", metadata={"source_type": "evidence", "pmid": "1"}, score=0.9)
    c2 = RetrievedChunk(text="b", metadata={"source_type": "evidence", "pmid": "2"}, score=0.3)

    with patch("src.rag.chain.classify_and_decompose", return_value=[{"question": "q", "type": "evidence_seeking"}]):
        with patch("src.rag.chain.retrieve", return_value=[c1, c2]):
            with patch(
                "src.rag.chain.decide_retrieval_state",
                return_value=RetrievalDecision(
                    state="answered", surviving_chunks=[c1], candidate_scores=[0.9, 0.3], floor=0.5, low_confidence_margin=0.05,
                ),
            ):
                decision_topk, _ = _retrieve_for_question("q", None, retrieval_mode="topk")
                decision_floor, _ = _retrieve_for_question("q", None, retrieval_mode="floor")

    assert [c.metadata["pmid"] for c in decision_topk.surviving_chunks] == ["1", "2"]
    assert [c.metadata["pmid"] for c in decision_floor.surviving_chunks] == ["1"]


def test_retrieve_for_question_invalid_mode_raises():
    from src.rag.chain import _retrieve_for_question

    with pytest.raises(ValueError):
        _retrieve_for_question("q", None, retrieval_mode="bogus")


def test_stream_answer_force_judge_true_bypasses_both_stages_and_marks_it():
    """
    force_judge is diagnostic-only. When True, neither real judge stage is
    called, and the emitted scope/evidence payloads carry bypassed=True so
    a record built from these events is never mistaken for a real verdict.
    """
    from unittest.mock import MagicMock

    from src.rag.types import RetrievalDecision, RetrievedChunk

    chunk = RetrievedChunk(
        text="Some evidence text.",
        metadata={"pmid": "123", "title": "t", "journal": "j", "year": "2024", "publication_type": "Review", "population": "type2"},
        score=0.9,
    )
    decision = RetrievalDecision(state="answered", surviving_chunks=[chunk], candidate_scores=[0.9], floor=0.5, low_confidence_margin=0.05)
    classified = [{"question": "q", "type": "evidence_seeking"}]

    mock_stream_cm = MagicMock()
    mock_stream_cm.__enter__.return_value.text_stream = iter(['[{"sentence": "Ok.", "chunk_ids": [1], "supported": true}]'])
    mock_stream_cm.__exit__.return_value = False

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-index"}):
        with patch("src.rag.chain._retrieve_for_question", return_value=(decision, classified)):
            with patch("src.rag.chain.judge_scope") as mock_scope:
                with patch("src.rag.chain.judge_evidence") as mock_evidence:
                    with patch("src.rag.chain.get_client") as mock_get_client:
                        mock_get_client.return_value.messages.stream.return_value = mock_stream_cm

                        from src.rag.chain import stream_answer

                        events = list(stream_answer("q", force_judge=True))

    mock_scope.assert_not_called()
    mock_evidence.assert_not_called()
    sources_event = next(e for e in events if e["event"] == "sources")
    assert sources_event["data"]["status"] == "answered"
    assert sources_event["data"]["scope"] == "in_charter"
    assert sources_event["data"]["evidence_verdict"] == "sufficient"
    assert sources_event["data"]["judge"]["bypassed"] is True


def test_stream_answer_force_judge_false_refuses_without_calling_real_stages():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-index"}):
        with patch("src.rag.chain._retrieve_for_question") as mock_retrieve:
            with patch("src.rag.chain.judge_scope") as mock_scope:
                with patch("src.rag.chain.judge_evidence") as mock_evidence:
                    with patch("src.rag.chain.get_client") as mock_get_client:
                        from src.rag.chain import stream_answer

                        events = list(stream_answer("q", force_judge=False))

    assert len(events) == 1
    assert events[0]["event"] == "refusal"
    data = events[0]["data"]
    assert data["status"] == "out_of_scope"
    assert data["scope"] == "unrelated"
    assert data["judge"]["bypassed"] is True
    mock_scope.assert_not_called()
    mock_evidence.assert_not_called()
    mock_retrieve.assert_not_called()
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
    classified = [{"question": "What does the evidence say about type 2 diabetes diet?", "type": "evidence_seeking"}]
    evidence_result = _evidence("sufficient", direct_map={"diet affects glycemic control": [1]})

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-index"}):
        with patch("src.rag.chain.judge_scope", return_value=_IN_CHARTER):
            with patch("src.rag.chain._retrieve_for_question", return_value=(decision, classified)):
                with patch("src.rag.chain.judge_evidence", return_value=evidence_result):
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
    assert "SOURCE TYPE: study" in context
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

    assert is_population_mismatch({"gestational"}, {"type2"}) is True
    assert is_population_mismatch({"type2"}, {"type2", "mixed"}) is False
    assert is_population_mismatch({"mixed"}, {"type2"}) is False
    assert is_population_mismatch({"type2"}, set()) is False


def test_population_mismatch_multi_population_covered_by_one():
    # "What is type 1 and type 2 diabetes" bug regression: requesting both
    # type1 and type2, with evidence covering only one of them, must NOT be
    # a mismatch as long as the evidence covers at least one named population
    # or the retrieved content is general ("mixed").
    from src.rag.chain import is_population_mismatch

    assert is_population_mismatch({"type1", "type2"}, {"type2"}) is False
    assert is_population_mismatch({"type1", "type2"}, {"mixed"}) is False


def test_population_mismatch_multi_population_covered_by_neither():
    from src.rag.chain import is_population_mismatch

    assert is_population_mismatch({"type1", "type2"}, {"gestational"}) is True


def test_filter_chunks_to_direct_keeps_only_direct_indices():
    from src.rag.chain import filter_chunks_to_direct
    from src.rag.types import RetrievedChunk

    chunks = [RetrievedChunk(text=f"c{i}", metadata={"pmid": str(i)}, score=0.9) for i in range(1, 4)]
    evidence_result = {"propositions": [
        {"claim": "a", "direct": [1], "partial": [2], "absent": False},
        {"claim": "b", "direct": [3], "partial": [], "absent": False},
    ]}
    filtered = filter_chunks_to_direct(chunks, evidence_result)
    assert [c.metadata["pmid"] for c in filtered] == ["1", "3"]


def test_filter_chunks_to_direct_falls_back_to_all_when_nothing_direct():
    """
    A "partial" verdict driven purely by a population mismatch (no
    proposition actually marked DIRECT) must not leave generation with zero
    context — fall back to the full retrieved set rather than an empty list.
    """
    from src.rag.chain import filter_chunks_to_direct
    from src.rag.types import RetrievedChunk

    chunks = [RetrievedChunk(text="c1", metadata={"pmid": "1"}, score=0.9)]
    evidence_result = {"propositions": [{"claim": "a", "direct": [], "partial": [1], "absent": False}]}
    assert filter_chunks_to_direct(chunks, evidence_result) == chunks


def test_status_instruction_partial_names_the_gap():
    from src.rag.chain import _status_instruction

    evidence_result = {"reason": "No excerpt reports a magnitude for weight loss."}
    text = _status_instruction("answered_partial", evidence_result)
    assert "PARTIAL" in text
    assert "No excerpt reports a magnitude for weight loss." in text


def test_status_instruction_adjacent_names_the_four_areas():
    from src.rag.chain import _status_instruction

    text = _status_instruction("answered_adjacent", {})
    assert "four core areas" in text


def test_status_instruction_answered_is_a_no_op():
    from src.rag.chain import _status_instruction

    assert "Not applicable" in _status_instruction("answered", {})


def test_evidence_audit_note_empty_when_no_mismatch():
    from src.rag.chain import _evidence_audit_note

    assert _evidence_audit_note({"population_mismatch": False}) == ""


def test_evidence_audit_note_names_both_populations_on_mismatch():
    from src.rag.chain import _evidence_audit_note

    note = _evidence_audit_note({
        "population_mismatch": True,
        "question_population": "type1",
        "evidence_populations": ["type2", "mixed"],
    })
    assert "type1" in note
    assert "type2, mixed" in note


def test_answered_partial_restricts_generation_to_direct_chunks():
    """
    Full flow for status=answered_partial: the model must only ever see
    the chunks the evidence audit marked DIRECT, not the full retrieved set
    — verified by checking which chunk texts made it into format_context's
    output (indirectly, via the "sources" event's chunk list).
    """
    from unittest.mock import MagicMock

    from src.rag.types import RetrievalDecision, RetrievedChunk

    direct_chunk = RetrievedChunk(text="Directly relevant excerpt.", metadata={"pmid": "1", "population": "type2"}, score=0.9)
    partial_chunk = RetrievedChunk(text="Only tangentially related excerpt.", metadata={"pmid": "2", "population": "type2"}, score=0.85)
    decision = RetrievalDecision(state="answered", surviving_chunks=[direct_chunk, partial_chunk], candidate_scores=[0.9, 0.85], floor=0.5, low_confidence_margin=0.05)
    classified = [{"question": "q", "type": "evidence_seeking"}]
    evidence_result = _evidence("partial", reason="Only the core claim is directly supported.", direct_map={"core claim": [1]})

    mock_stream_cm = MagicMock()
    mock_stream_cm.__enter__.return_value.text_stream = iter(['[{"sentence": "Ok.", "chunk_ids": [1], "supported": true}]'])
    mock_stream_cm.__exit__.return_value = False

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-index"}):
        with patch("src.rag.chain.judge_scope", return_value=_IN_CHARTER):
            with patch("src.rag.chain._retrieve_for_question", return_value=(decision, classified)):
                with patch("src.rag.chain.judge_evidence", return_value=evidence_result):
                    with patch("src.rag.chain.get_client") as mock_get_client:
                        mock_get_client.return_value.messages.stream.return_value = mock_stream_cm

                        from src.rag.chain import stream_answer

                        events = list(stream_answer("q"))

    sources_event = next(e for e in events if e["event"] == "sources")
    assert sources_event["data"]["status"] == "answered_partial"
    assert sources_event["data"]["scope_model"] == "m"  # Stage A's model string is never dropped on the success path either
    pmids_sent = [s["pmid"] for s in sources_event["data"]["sources"]]
    assert pmids_sent == ["1"]  # only the DIRECT chunk, never the partial one

    # The system prompt actually sent to the model must name the gap.
    call_kwargs = mock_get_client.return_value.messages.stream.call_args.kwargs
    assert "PARTIAL" in call_kwargs["system"]
    assert "Only the core claim is directly supported." in call_kwargs["system"]


def test_answered_adjacent_generates_with_scope_caveat():
    from unittest.mock import MagicMock

    from src.rag.types import RetrievalDecision, RetrievedChunk

    chunk = RetrievedChunk(text="Retinopathy screening interval is 1-2 years.", metadata={"pmid": "1", "population": "type2"}, score=0.8)
    decision = RetrievalDecision(state="answered", surviving_chunks=[chunk], candidate_scores=[0.8], floor=0.5, low_confidence_margin=0.05)
    classified = [{"question": "q", "type": "evidence_seeking"}]
    evidence_result = _evidence("sufficient", direct_map={"screening interval": [1]})

    mock_stream_cm = MagicMock()
    mock_stream_cm.__enter__.return_value.text_stream = iter(['[{"sentence": "Ok.", "chunk_ids": [1], "supported": true}]'])
    mock_stream_cm.__exit__.return_value = False

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-index"}):
        with patch("src.rag.chain.judge_scope", return_value=_ADJACENT):
            with patch("src.rag.chain._retrieve_for_question", return_value=(decision, classified)):
                with patch("src.rag.chain.judge_evidence", return_value=evidence_result):
                    with patch("src.rag.chain.get_client") as mock_get_client:
                        mock_get_client.return_value.messages.stream.return_value = mock_stream_cm

                        from src.rag.chain import stream_answer

                        events = list(stream_answer("q"))

    sources_event = next(e for e in events if e["event"] == "sources")
    assert sources_event["data"]["status"] == "answered_adjacent"
    assert sources_event["data"]["scope"] == "adjacent"
    call_kwargs = mock_get_client.return_value.messages.stream.call_args.kwargs
    assert "four core areas" in call_kwargs["system"]


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


def test_decide_retrieval_state_per_tier_floor_lets_background_survive_below_evidence_floor():
    """
    Phase 4: background_floor lets a lower-scoring background chunk clear
    retrieval even when the evidence floor wouldn't have let it through —
    the two tiers embed with different score distributions and are tuned
    independently.
    """
    from src.rag.retriever import decide_retrieval_state
    from src.rag.types import RetrievedChunk

    candidates = [
        RetrievedChunk(text="e", metadata={"source_type": "evidence"}, score=0.55),
        RetrievedChunk(text="b", metadata={"source_type": "background"}, score=0.35),
    ]
    decision = decide_retrieval_state(candidates, floor=0.5, low_confidence_margin=0.05, background_floor=0.30)

    assert decision.state == "answered"
    assert len(decision.surviving_chunks) == 2
    assert decision.floor == 0.5
    assert decision.background_floor == 0.30


def test_decide_retrieval_state_per_tier_floor_drops_background_below_its_own_floor():
    from src.rag.retriever import decide_retrieval_state
    from src.rag.types import RetrievedChunk

    candidates = [
        RetrievedChunk(text="e", metadata={"source_type": "evidence"}, score=0.55),
        RetrievedChunk(text="b", metadata={"source_type": "background"}, score=0.35),
    ]
    decision = decide_retrieval_state(candidates, floor=0.5, low_confidence_margin=0.05, background_floor=0.40)

    assert len(decision.surviving_chunks) == 1
    assert decision.surviving_chunks[0].metadata["source_type"] == "evidence"


def test_decide_retrieval_state_background_floor_none_matches_uniform_floor():
    """background_floor=None (the default) preserves the original single-floor behavior."""
    from src.rag.retriever import decide_retrieval_state
    from src.rag.types import RetrievedChunk

    candidates = [
        RetrievedChunk(text="e", metadata={"source_type": "evidence"}, score=0.55),
        RetrievedChunk(text="b", metadata={"source_type": "background"}, score=0.35),
    ]
    decision = decide_retrieval_state(candidates, floor=0.5, low_confidence_margin=0.05)

    assert len(decision.surviving_chunks) == 1
    assert decision.background_floor is None


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
        "sentence": "x", "chunk_ids": [1, 2], "supported": True, "new_paragraph": False,
    }
    # Missing supported -> derived from chunk_ids presence.
    assert _validate_sentence({"sentence": "x", "chunk_ids": [1]})["supported"] is True
    assert _validate_sentence({"sentence": "x", "chunk_ids": []})["supported"] is False
    # Non-list chunk_ids -> coerced to [].
    assert _validate_sentence({"sentence": "x", "chunk_ids": "bad"})["chunk_ids"] == []
    # new_paragraph: present and true -> kept; missing/invalid -> False.
    assert _validate_sentence({"sentence": "x", "new_paragraph": True})["new_paragraph"] is True
    assert _validate_sentence({"sentence": "x", "new_paragraph": "yes"})["new_paragraph"] is False
    assert _validate_sentence({"sentence": "x"})["new_paragraph"] is False
    # No sentence text at all -> rejected.
    assert _validate_sentence({"chunk_ids": [1]}) is None
    assert _validate_sentence({"sentence": "   "}) is None
    assert _validate_sentence("not a dict") is None


def test_compute_groundedness():
    from src.rag.chain import compute_groundedness
    from src.rag.types import RetrievedChunk

    chunks = [
        RetrievedChunk(text="e1", metadata={"source_type": "evidence"}, score=0.9),
        RetrievedChunk(text="e2", metadata={"source_type": "evidence"}, score=0.8),
        RetrievedChunk(text="e3", metadata={"source_type": "evidence"}, score=0.7),
    ]
    sentences = [
        {"sentence": "a", "chunk_ids": [1], "supported": True},
        {"sentence": "b", "chunk_ids": [], "supported": False},
        {"sentence": "c", "chunk_ids": [2, 3], "supported": True},
    ]
    result = compute_groundedness(sentences, chunks)
    assert result["supported_sentences"] == 2
    assert result["total_sentences"] == 3
    assert result["groundedness"] == round(2 / 3, 4)

    assert compute_groundedness([], chunks)["groundedness"] is None


def test_compute_groundedness_splits_evidence_and_background():
    """
    Overall groundedness alone would let a purely definitional,
    background-only answer look just as "grounded" as one backed by real
    studies — the split is what actually answers "is there a study behind
    this," per rule 2.
    """
    from src.rag.chain import compute_groundedness
    from src.rag.types import RetrievedChunk

    chunks = [
        RetrievedChunk(text="e1", metadata={"source_type": "evidence"}, score=0.9),
        RetrievedChunk(text="b1", metadata={"source_type": "background"}, score=0.6),
    ]
    sentences = [
        {"sentence": "Definition.", "chunk_ids": [2], "supported": True},
        {"sentence": "Research finding.", "chunk_ids": [1], "supported": True},
        {"sentence": "Unsupported transition.", "chunk_ids": [], "supported": False},
    ]
    result = compute_groundedness(sentences, chunks)
    assert result["evidence_supported_sentences"] == 1
    assert result["background_supported_sentences"] == 1
    assert result["supported_sentences"] == 2
    assert result["groundedness_evidence"] == round(1 / 3, 4)
    assert result["groundedness_background"] == round(1 / 3, 4)
    assert result["groundedness"] == round(2 / 3, 4)


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


def test_source_type_for_indices():
    from src.rag.chain import _source_type_for_indices, _sentence_payload
    from src.rag.types import RetrievedChunk

    chunks = [
        RetrievedChunk(text="e", metadata={"source_type": "evidence"}, score=0.9),
        RetrievedChunk(text="b", metadata={"source_type": "background"}, score=0.6),
    ]
    assert _source_type_for_indices([1], chunks) == "evidence"
    assert _source_type_for_indices([2], chunks) == "background"
    assert _source_type_for_indices([1, 2], chunks) == "evidence"  # evidence wins if mixed
    assert _source_type_for_indices([], chunks) is None
    assert _source_type_for_indices([99], chunks) is None  # out-of-range index

    payload = _sentence_payload({"sentence": "x", "chunk_ids": [2], "supported": True}, chunks)
    assert payload["source_type"] == "background"


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
    classified = [{"question": "Does Diet A improve HbA1c in type 2 diabetes?", "type": "evidence_seeking"}]

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

    evidence_result = _evidence("sufficient", direct_map={"Diet A improves HbA1c": [1, 2]})

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-index"}):
        with patch("src.rag.chain.judge_scope", return_value=_IN_CHARTER):
            with patch("src.rag.chain._retrieve_for_question", return_value=(decision, classified)):
                with patch("src.rag.chain.judge_evidence", return_value=evidence_result):
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
    The model is always expected to use the JSON array format now. If it
    ever ignores that instruction and emits plain text anyway, the stream
    must still degrade into the documented {sentence, chunk_ids, supported}
    shape rather than losing the response. Unlike the retired
    determine_final_state() behavior, status is fixed by the pre-generation
    scope/evidence decision and is never demoted post-hoc just because the
    model didn't cite anything — that's now purely a groundedness/faithfulness
    signal, not a status change.
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
    classified = [{"question": "An unanswerable question.", "type": "evidence_seeking"}]
    evidence_result = _evidence("sufficient", direct_map={"the question": [1]})

    mock_stream_cm = MagicMock()
    mock_stream_cm.__enter__.return_value.text_stream = iter([plain_text_response])
    mock_stream_cm.__exit__.return_value = False

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-index"}):
        with patch("src.rag.chain.judge_scope", return_value=_IN_CHARTER):
            with patch("src.rag.chain._retrieve_for_question", return_value=(decision, classified)):
                with patch("src.rag.chain.judge_evidence", return_value=evidence_result):
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
    assert done_data["status"] == "answered"
    assert done_data["state"] == "answered"


def test_judge_parses_clean_json():
    from src.rag.judge import _parse_judge_response

    result = _parse_judge_response('{"answerable": true, "reason": "The excerpts directly address the question."}')
    assert result == {"answerable": True, "reason": "The excerpts directly address the question."}


def test_judge_strips_code_fences():
    from src.rag.judge import _parse_judge_response

    raw = '```json\n{"answerable": false, "reason": "Off-topic."}\n```'
    result = _parse_judge_response(raw)
    assert result == {"answerable": False, "reason": "Off-topic."}


def test_judge_strips_bare_code_fences_no_language_tag():
    from src.rag.judge import _parse_judge_response

    raw = '```\n{"answerable": true, "reason": "ok"}\n```'
    result = _parse_judge_response(raw)
    assert result == {"answerable": True, "reason": "ok"}


def test_judge_malformed_json_defaults_to_not_answerable():
    from src.rag.judge import _parse_judge_response

    result = _parse_judge_response("this is not json at all")
    assert result["answerable"] is False
    assert "reason" in result


def test_judge_missing_answerable_field_defaults_to_not_answerable():
    from src.rag.judge import _parse_judge_response

    result = _parse_judge_response('{"reason": "no verdict given"}')
    assert result["answerable"] is False


def test_judge_non_bool_answerable_defaults_to_not_answerable():
    from src.rag.judge import _parse_judge_response

    result = _parse_judge_response('{"answerable": "yes", "reason": "should have been a bool"}')
    assert result["answerable"] is False


def test_judge_missing_reason_defaults_to_placeholder():
    from src.rag.judge import _parse_judge_response

    result = _parse_judge_response('{"answerable": true}')
    assert result["answerable"] is True
    assert result["reason"] == "No reason given."


def test_judge_answerability_no_chunks_short_circuits_without_api_call():
    from unittest.mock import MagicMock

    from src.rag.judge import judge_answerability

    mock_client = MagicMock()
    result = judge_answerability(mock_client, "Some question?", [])
    assert result["answerable"] is False
    mock_client.messages.create.assert_not_called()


def test_judge_answerability_calls_model_and_parses_response():
    from unittest.mock import MagicMock

    from src.rag.judge import judge_answerability
    from src.rag.types import RetrievedChunk

    chunk = RetrievedChunk(
        text="Low-carbohydrate diets improved HbA1c.",
        metadata={"pmid": "123", "title": "t", "source_type": "evidence"},
        score=0.8,
    )

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = '{"answerable": true, "reason": "Directly relevant."}'
    mock_response = MagicMock()
    mock_response.content = [text_block]
    mock_response.model = "claude-haiku-4-5-20251001"
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    result = judge_answerability(mock_client, "Does diet improve HbA1c?", [chunk])

    assert result == {"answerable": True, "reason": "Directly relevant.", "model": "claude-haiku-4-5-20251001"}
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-haiku-4-5"
    assert "PMID 123" in call_kwargs["messages"][0]["content"]


def test_judge_answerability_api_error_defaults_to_not_answerable():
    from unittest.mock import MagicMock

    import anthropic
    import httpx2

    from src.rag.judge import judge_answerability
    from src.rag.types import RetrievedChunk

    chunk = RetrievedChunk(text="Some excerpt.", metadata={"pmid": "123"}, score=0.8)
    fake_request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = anthropic.APIConnectionError(request=fake_request)

    result = judge_answerability(mock_client, "A question?", [chunk])

    assert result["answerable"] is False


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
    assert "sentence" in event_types
    assert event_types[-1] == "done"
