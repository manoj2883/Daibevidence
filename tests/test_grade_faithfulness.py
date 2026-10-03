"""Unit tests for the independent faithfulness grader (scripts.grade_faithfulness)."""
import json
from unittest.mock import MagicMock

import anthropic
import httpx2


def _mock_response(text, model="claude-sonnet-5-20260101"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    resp.model = model
    return resp


def test_grade_answer_uses_the_generation_model_not_the_judge_model():
    """
    Step 8's independence requirement: the grader must call
    src.rag.chain.get_model() (Sonnet), never src.rag.judge.get_judge_model()
    (Haiku) — otherwise it shares a model with Stage A/B.
    """
    from scripts.grade_faithfulness import grade_answer

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response(json.dumps({
        "sentence_grades": [{"index": 1, "label": "SUPPORTED", "why": "ok"}],
        "answers_the_question": 1, "overall": "SUPPORTED",
    }))
    grade_answer(mock_client, "q?", [{"source_type": "evidence", "pmid": "1", "title": "t", "excerpt": "e"}],
                 [{"sentence": "s", "chunk_ids": [1]}])

    call_kwargs = mock_client.messages.create.call_args.kwargs
    from src.rag.chain import get_model
    assert call_kwargs["model"] == get_model()


def test_grader_prompt_shares_no_stage_b_vocabulary():
    """
    Step 8: the grader prompt must share no phrasing with Stage B's prompt
    — spot-check the distinctive terms Stage B uses that would indicate
    copy-paste reuse rather than an independently-worded rubric.
    """
    from scripts.grade_faithfulness import GRADER_SYSTEM_PROMPT
    from src.rag.evidence_judge import EVIDENCE_SYSTEM_PROMPT_TEMPLATE

    stage_b_distinctive_terms = ["DECOMPOSE", "AUDIT.", "DIRECT:", "PARTIAL:", "ABSENT:", "Topical similarity"]
    for term in stage_b_distinctive_terms:
        assert term not in GRADER_SYSTEM_PROMPT, f"grader prompt reuses Stage B phrasing: {term!r}"
        assert term in EVIDENCE_SYSTEM_PROMPT_TEMPLATE, f"test fixture out of date: {term!r} not even in Stage B prompt"


def test_grade_answer_parses_clean_response():
    from scripts.grade_faithfulness import grade_answer

    mock_client = MagicMock()
    payload = {
        "sentence_grades": [{"index": 1, "label": "SUPPORTED", "why": "Directly stated."}],
        "answers_the_question": 1,
        "overall": "SUPPORTED",
    }
    mock_client.messages.create.return_value = _mock_response(json.dumps(payload))
    result = grade_answer(mock_client, "q?", [{"excerpt": "e"}], [{"sentence": "s", "chunk_ids": [1]}])
    assert result["overall"] == "SUPPORTED"
    assert result["sentence_grades"][0]["label"] == "SUPPORTED"


def test_grade_answer_no_sentences_short_circuits():
    from scripts.grade_faithfulness import grade_answer

    mock_client = MagicMock()
    result = grade_answer(mock_client, "q?", [], [])
    assert result["overall"] == "UNSUPPORTED"
    mock_client.messages.create.assert_not_called()


def test_grade_answer_malformed_json_fails_closed():
    from scripts.grade_faithfulness import grade_answer

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response("not json")
    result = grade_answer(mock_client, "q?", [{"excerpt": "e"}], [{"sentence": "s", "chunk_ids": [1]}])
    assert result["overall"] == "UNSUPPORTED"
    assert "grading_error" in result


def test_grade_answer_invalid_overall_enum_fails_closed():
    from scripts.grade_faithfulness import grade_answer

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response(json.dumps({
        "sentence_grades": [], "answers_the_question": None, "overall": "MOSTLY_FINE",
    }))
    result = grade_answer(mock_client, "q?", [{"excerpt": "e"}], [{"sentence": "s", "chunk_ids": [1]}])
    assert result["overall"] == "UNSUPPORTED"


def test_grade_answer_api_error_fails_closed():
    from scripts.grade_faithfulness import grade_answer

    mock_client = MagicMock()
    fake_request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    mock_client.messages.create.side_effect = anthropic.APIConnectionError(request=fake_request)
    result = grade_answer(mock_client, "q?", [{"excerpt": "e"}], [{"sentence": "s", "chunk_ids": [1]}])
    assert result["overall"] == "UNSUPPORTED"


def test_cited_chunk_texts_only_includes_cited_indices():
    from scripts.grade_faithfulness import _cited_chunk_texts

    sources = [{"excerpt": "first"}, {"excerpt": "second"}, {"excerpt": "third"}]
    sentence_records = [{"chunk_ids": [1]}, {"chunk_ids": [3]}, {"chunk_ids": []}]
    result = _cited_chunk_texts(sources, sentence_records)
    assert "first" in result
    assert "third" in result
    assert "second" not in result


def test_write_hand_grading_csv_has_empty_human_grade_column(tmp_path):
    from scripts.grade_faithfulness import write_hand_grading_csv
    import csv as csv_module

    rows = [{
        "id": 1, "question": "q?", "generated_answer": "a.",
        "cited_chunk_texts": "[1] e", "verdict": {"overall": "SUPPORTED"},
    }]
    out_path = tmp_path / "hand_grading.csv"
    write_hand_grading_csv(str(out_path), rows)

    with open(out_path, newline="", encoding="utf-8") as f:
        reader = list(csv_module.DictReader(f))
    assert reader[0]["human_grade"] == ""
    assert reader[0]["automated_grade"] == "SUPPORTED"
    assert reader[0]["id"] == "1"


def test_partial_rubric_shares_no_stage_b_vocabulary():
    from scripts.grade_faithfulness import PARTIAL_RUBRIC

    for term in ["DECOMPOSE", "AUDIT.", "DIRECT:", "PARTIAL:", "ABSENT:", "Topical similarity"]:
        assert term not in PARTIAL_RUBRIC


def test_partial_overall_rule():
    from scripts.grade_faithfulness import partial_overall

    g = lambda *labels: [{"index": i, "label": l} for i, l in enumerate(labels, 1)]
    # every factual sentence supported + gap stated -> SUPPORTED, gap sentence itself is NOT_APPLICABLE
    assert partial_overall(g("SUPPORTED", "SUPPORTED", "NOT_APPLICABLE"), True) == "SUPPORTED"
    # missing gap statement is not "fully supported"
    assert partial_overall(g("SUPPORTED"), False) == "PARTIALLY_SUPPORTED"
    assert partial_overall(g("SUPPORTED", "PARTIALLY_SUPPORTED"), True) == "PARTIALLY_SUPPORTED"
    assert partial_overall(g("SUPPORTED", "UNSUPPORTED"), True) == "UNSUPPORTED"
    # nothing factual at all -> nothing was answered
    assert partial_overall(g("NOT_APPLICABLE"), True) == "UNSUPPORTED"


def test_partial_answers_use_the_partial_rubric_and_code_computed_overall():
    from scripts.grade_faithfulness import PARTIAL_RUBRIC, grade_answer

    mock_client = MagicMock()
    # The model's own overall penalizes missing coverage; the partial rule must override it.
    mock_client.messages.create.return_value = _mock_response(json.dumps({
        "sentence_grades": [{"index": 1, "label": "SUPPORTED", "why": "ok"}, {"index": 2, "label": "NOT_APPLICABLE", "why": "gap"}],
        "answers_the_question": None, "overall": "UNSUPPORTED", "states_gap": True,
    }))
    result = grade_answer(mock_client, "q?", [{"excerpt": "e"}],
                          [{"sentence": "s", "chunk_ids": [1]}, {"sentence": "The studies found here don't address x.", "chunk_ids": []}],
                          status="answered_partial")
    assert mock_client.messages.create.call_args.kwargs["system"].endswith(PARTIAL_RUBRIC)
    assert result["overall"] == "SUPPORTED" and result["model_overall"] == "UNSUPPORTED" and result["rubric"] == "partial_v2"


def test_answered_status_keeps_the_original_rubric():
    from scripts.grade_faithfulness import GRADER_SYSTEM_PROMPT, grade_answer

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response(json.dumps({
        "sentence_grades": [{"index": 1, "label": "SUPPORTED", "why": "ok"}], "answers_the_question": 1, "overall": "SUPPORTED",
    }))
    grade_answer(mock_client, "q?", [{"excerpt": "e"}], [{"sentence": "s", "chunk_ids": [1]}], status="answered")
    assert mock_client.messages.create.call_args.kwargs["system"] == GRADER_SYSTEM_PROMPT


def test_partial_grade_without_states_gap_fails_closed():
    from scripts.grade_faithfulness import grade_answer

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response(json.dumps({
        "sentence_grades": [{"index": 1, "label": "SUPPORTED", "why": "ok"}], "answers_the_question": 1, "overall": "SUPPORTED",
    }))
    result = grade_answer(mock_client, "q?", [{"excerpt": "e"}], [{"sentence": "s", "chunk_ids": [1]}], status="answered_partial")
    assert result["overall"] == "UNSUPPORTED" and "states_gap" in result["grading_error"]
