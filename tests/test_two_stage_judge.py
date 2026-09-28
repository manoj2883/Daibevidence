"""
Unit tests for the two-stage answerability judge (src.rag.scope_judge,
src.rag.evidence_judge) — no live API calls, everything mocked, matching the
project's existing test style (see tests/test_rag.py).
"""
import json
import os
from unittest.mock import MagicMock, patch

import anthropic
import httpx2
import pytest


# ---------- Stage A: scope judge ----------

def _mock_response(text, model="claude-haiku-4-5-20251001"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    resp.model = model
    return resp


def test_load_charter_reads_real_file():
    from src.rag.scope_judge import load_charter

    text = load_charter()
    assert "in_charter" in text or "In charter" in text
    assert "diet_nutrition" in text or "Diet and nutrition" in text


def test_judge_scope_parses_valid_response_and_caches():
    from src.rag.scope_judge import get_scope_cache, judge_scope

    get_scope_cache().clear()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response(
        json.dumps({"scope": "in_charter", "area": "diet_nutrition", "personal_dosing": False, "reason": "Asks about diet and T2D."})
    )

    result = judge_scope(mock_client, "Does a Mediterranean diet help type 2 diabetes?")
    assert result["scope"] == "in_charter"
    assert result["topic"] == "diet_nutrition"
    assert result["cached"] is False
    assert mock_client.messages.create.call_count == 1

    # Second call with the same (normalized) question hits the cache, no new API call.
    result2 = judge_scope(mock_client, "  Does a Mediterranean diet help type 2 diabetes?  ")
    assert result2["cached"] is True
    assert mock_client.messages.create.call_count == 1


def test_judge_scope_adjacent_with_no_topic():
    from src.rag.scope_judge import get_scope_cache, judge_scope

    get_scope_cache().clear()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response(
        json.dumps({"scope": "adjacent", "area": None, "personal_dosing": False, "reason": "Asks about insulin dosing, not the four core areas."})
    )
    result = judge_scope(mock_client, "What insulin-to-carb ratio should I use?")
    assert result["scope"] == "adjacent"
    assert result["topic"] is None


def test_judge_scope_invalid_topic_for_non_in_charter_is_nulled():
    from src.rag.scope_judge import get_scope_cache, judge_scope

    get_scope_cache().clear()
    mock_client = MagicMock()
    # A model that incorrectly sets a topic on a non-in_charter scope should
    # not propagate that topic — it's only meaningful for in_charter.
    mock_client.messages.create.return_value = _mock_response(
        json.dumps({"scope": "unrelated", "area": "diet_nutrition", "personal_dosing": False, "reason": "Not about diabetes at all."})
    )
    result = judge_scope(mock_client, "What is the capital of Australia?")
    assert result["scope"] == "unrelated"
    assert result["topic"] is None


def test_judge_scope_retries_once_then_fails_closed_to_unrelated():
    from src.rag.scope_judge import get_scope_cache, judge_scope

    get_scope_cache().clear()
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        _mock_response("not json at all"),
        _mock_response("still not json"),
    ]
    result = judge_scope(mock_client, "Some question?")
    assert result["scope"] == "unrelated"
    assert mock_client.messages.create.call_count == 2


def test_judge_scope_second_attempt_succeeds_after_first_failure():
    from src.rag.scope_judge import get_scope_cache, judge_scope

    get_scope_cache().clear()
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        _mock_response("garbage"),
        _mock_response(json.dumps({"scope": "in_charter", "area": "glycemic_control", "personal_dosing": False, "reason": "ok"})),
    ]
    result = judge_scope(mock_client, "What HbA1c target should I aim for?")
    assert result["scope"] == "in_charter"
    assert result["topic"] == "glycemic_control"
    assert mock_client.messages.create.call_count == 2


def test_judge_scope_api_error_fails_closed():
    from src.rag.scope_judge import get_scope_cache, judge_scope

    get_scope_cache().clear()
    mock_client = MagicMock()
    fake_request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    mock_client.messages.create.side_effect = anthropic.APIConnectionError(request=fake_request)
    result = judge_scope(mock_client, "Some question?")
    assert result["scope"] == "unrelated"


def test_judge_scope_invalid_scope_value_rejected():
    from src.rag.scope_judge import get_scope_cache, judge_scope

    get_scope_cache().clear()
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        _mock_response(json.dumps({"scope": "maybe", "area": None, "personal_dosing": False, "reason": "?"})),
        _mock_response(json.dumps({"scope": "adjacent", "area": None, "personal_dosing": False, "reason": "retry ok"})),
    ]
    result = judge_scope(mock_client, "Some question?")
    assert result["scope"] == "adjacent"
    assert mock_client.messages.create.call_count == 2


# ---------- Stage B: evidence judge ----------

def _chunk(text, population="type2", year="2024"):
    from src.rag.types import RetrievedChunk

    return RetrievedChunk(text=text, metadata={"population": population, "year": year}, score=0.8)


def test_judge_evidence_parses_sufficient_verdict():
    from src.rag.evidence_judge import judge_evidence

    mock_client = MagicMock()
    payload = {
        "propositions": [{"claim": "low-carb reduces HbA1c", "core": True, "direct": [1], "partial": [], "absent": False}],
        "on_topic": True,
        "cross_study_comparison": False,
        "question_population": "type2",
        "evidence_populations": ["type2"],
        "population_mismatch": False,
        "verdict": "sufficient",
        "reason": "Direct evidence found.",
    }
    mock_client.messages.create.return_value = _mock_response(json.dumps(payload))
    result = judge_evidence(mock_client, "Does low-carb reduce HbA1c?", [_chunk("Low-carb reduced HbA1c by 0.5%.")])
    assert result["verdict"] == "sufficient"
    assert result["propositions"][0]["direct"] == [1]
    assert result["population_mismatch"] is False


def test_judge_evidence_no_chunks_short_circuits_to_insufficient():
    from src.rag.evidence_judge import judge_evidence

    mock_client = MagicMock()
    result = judge_evidence(mock_client, "Some question?", [])
    assert result["verdict"] == "insufficient"
    mock_client.messages.create.assert_not_called()


def test_judge_evidence_malformed_json_retries_then_fails_closed():
    from src.rag.evidence_judge import judge_evidence

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        _mock_response("not json"),
        _mock_response("also not json"),
    ]
    result = judge_evidence(mock_client, "Some question?", [_chunk("some text")])
    assert result["verdict"] == "insufficient"
    assert mock_client.messages.create.call_count == 2


def test_judge_evidence_missing_verdict_field_fails_schema_then_retries():
    from src.rag.evidence_judge import judge_evidence

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        _mock_response(json.dumps({"propositions": [], "reason": "no verdict key"})),
        _mock_response(json.dumps({
            "propositions": [{"claim": "c", "core": True, "direct": [1], "partial": [], "absent": False}],
            "on_topic": True, "cross_study_comparison": False,
            "question_population": None, "evidence_populations": [],
            "population_mismatch": False, "verdict": "partial", "reason": "ok now",
        })),
    ]
    result = judge_evidence(mock_client, "Some question?", [_chunk("some text")])
    assert result["verdict"] == "partial"
    assert mock_client.messages.create.call_count == 2


def test_judge_evidence_invalid_verdict_enum_rejected():
    from src.rag.evidence_judge import judge_evidence

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response(json.dumps({
        "propositions": [], "question_population": None, "evidence_populations": [],
        "population_mismatch": False, "verdict": "yes", "reason": "bad enum value",
    }))
    result = judge_evidence(mock_client, "Some question?", [_chunk("some text")])
    assert result["verdict"] == "insufficient"


def test_judge_evidence_api_error_fails_closed():
    from src.rag.evidence_judge import judge_evidence

    mock_client = MagicMock()
    fake_request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    mock_client.messages.create.side_effect = anthropic.APIConnectionError(request=fake_request)
    result = judge_evidence(mock_client, "Some question?", [_chunk("some text")])
    assert result["verdict"] == "insufficient"


def test_judge_evidence_prompt_excludes_similarity_scores():
    """
    Step 4 explicitly requires no similarity scores in the Stage B prompt —
    verify the passages block never renders a chunk's .score.
    """
    from src.rag.evidence_judge import _format_passages

    chunk = _chunk("some evidence text")
    chunk.score = 0.987654
    rendered = _format_passages([chunk])
    assert "0.987654" not in rendered
    assert "0.98" not in rendered
    assert "[C1]" in rendered
    assert "population=type2" in rendered
    assert "year=2024" in rendered


def test_judge_evidence_normalizes_c_prefixed_chunk_refs():
    """
    Live testing found the model sometimes returns direct/partial entries
    as "C1" (matching the "[C{n}]" passage label) rather than a bare int —
    both are legitimate outputs of the same verbatim prompt, since it never
    specifies which. Both must normalize to the same plain int.
    """
    from src.rag.evidence_judge import judge_evidence

    mock_client = MagicMock()
    payload = {
        "propositions": [
            {"claim": "a", "core": True, "direct": ["C1", "c3"], "partial": [2], "absent": False},
            {"claim": "b", "core": False, "direct": [], "partial": ["C4"], "absent": False},
        ],
        "on_topic": True,
        "cross_study_comparison": False,
        "question_population": "type2",
        "evidence_populations": ["type2"],
        "population_mismatch": False,
        "verdict": "partial",
        "reason": "ok",
    }
    mock_client.messages.create.return_value = _mock_response(json.dumps(payload))
    result = judge_evidence(mock_client, "q?", [_chunk("t1"), _chunk("t2"), _chunk("t3"), _chunk("t4")])

    assert result["propositions"][0]["direct"] == [1, 3]
    assert result["propositions"][0]["partial"] == [2]
    assert result["propositions"][1]["partial"] == [4]


def test_normalize_chunk_ref_rejects_garbage():
    from src.rag.evidence_judge import _normalize_chunk_ref

    assert _normalize_chunk_ref("C1") == 1
    assert _normalize_chunk_ref("c12") == 12
    assert _normalize_chunk_ref(3) == 3
    assert _normalize_chunk_ref(True) is None  # bool is an int subclass in Python — must not sneak through
    assert _normalize_chunk_ref("excerpt 1") is None
    assert _normalize_chunk_ref(None) is None


# ---------- Revised prompts (2026-09-28): new fields and validation ----------

def _valid_evidence_payload(**overrides):
    payload = {
        "propositions": [
            {"claim": "core claim", "core": True, "direct": [1], "partial": [], "absent": False},
            {"claim": "secondary", "core": False, "direct": [], "partial": [1], "absent": False},
        ],
        "on_topic": True,
        "question_population": "type 2 diabetes",
        "evidence_populations": ["type 2 diabetes"],
        "population_mismatch": False,
        "cross_study_comparison": True,
        "verdict": "partial",
        "reason": "Comparison assembled across studies.",
    }
    payload.update(overrides)
    return payload


def test_scope_prompt_loads_areas_and_adjacent_examples_from_charter():
    from src.rag.scope_judge import build_scope_prompt, load_charter

    prompt = build_scope_prompt()
    charter = load_charter()
    for area in ("glycemic_control", "body_composition_weight", "diet_nutrition", "diet_medication_interaction"):
        assert f"- {area}:" in prompt
        assert f"- {area}:" in charter
    assert "Examples: eye, nerve, kidney, or foot complications; mental health; devices; cost or access to care." in prompt
    assert "{areas}" not in prompt and "{adjacent_examples}" not in prompt


def test_scope_prompt_source_does_not_duplicate_charter_areas():
    import inspect

    import src.rag.scope_judge as scope_judge

    # The area descriptions live only in the charter; the module must not hardcode them.
    assert "fat distribution, lean or muscle mass" not in inspect.getsource(scope_judge)


def test_scope_prompt_fails_loudly_when_charter_lacks_an_area(monkeypatch):
    import src.rag.scope_judge as scope_judge

    charter = scope_judge.load_charter().replace("- diet_nutrition:", "- diet:")
    monkeypatch.setattr(scope_judge, "_charter_text", charter)
    with pytest.raises(ValueError, match="diet_nutrition"):
        scope_judge.build_scope_prompt()


def test_judge_scope_returns_new_fields_with_topic_alias():
    from src.rag.scope_judge import get_scope_cache, judge_scope

    get_scope_cache().clear()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response(json.dumps({
        "outcome": "lean muscle mass", "area": "body_composition_weight", "scope": "in_charter",
        "personal_dosing": False, "reason": "Outcome is lean mass.",
    }))
    result = judge_scope(mock_client, "Do GLP-1 drugs cause loss of lean mass?")
    assert result["area"] == "body_composition_weight"
    assert result["topic"] == "body_composition_weight"
    assert result["outcome"] == "lean muscle mass"
    assert result["personal_dosing"] is False


def test_judge_scope_missing_personal_dosing_is_a_schema_failure():
    from src.rag.scope_judge import get_scope_cache, judge_scope

    get_scope_cache().clear()
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        _mock_response(json.dumps({"scope": "in_charter", "area": "glycemic_control", "reason": "no flag"})),
        _mock_response(json.dumps({"scope": "in_charter", "area": "glycemic_control", "personal_dosing": True, "reason": "ok"})),
    ]
    result = judge_scope(mock_client, "How much metformin should I take?")
    assert result["personal_dosing"] is True
    assert mock_client.messages.create.call_count == 2


def test_judge_scope_max_tokens_stop_is_treated_as_parse_failure():
    from src.rag.scope_judge import get_scope_cache, judge_scope

    get_scope_cache().clear()
    truncated = _mock_response('{"scope": "in_charter", "area": "glycemic_control", "personal_dosing": false, "reason": "ok"}')
    truncated.stop_reason = "max_tokens"
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [truncated, truncated]
    result = judge_scope(mock_client, "Some question?")
    assert result["scope"] == "unrelated"  # failed closed after retry
    assert mock_client.messages.create.call_count == 2


def test_judge_evidence_returns_on_topic_and_cross_study_comparison():
    from src.rag.evidence_judge import judge_evidence

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response(json.dumps(_valid_evidence_payload()))
    result = judge_evidence(mock_client, "Is X better than Y?", [_chunk("t1")])
    assert result["on_topic"] is True
    assert result["cross_study_comparison"] is True
    assert [p["core"] for p in result["propositions"]] == [True, False]


@pytest.mark.parametrize("cores", [[False, False], [True, True]])
def test_judge_evidence_requires_exactly_one_core_proposition(cores):
    from src.rag.evidence_judge import judge_evidence

    bad = _valid_evidence_payload()
    for prop, core in zip(bad["propositions"], cores):
        prop["core"] = core
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [_mock_response(json.dumps(bad)), _mock_response(json.dumps(bad))]
    result = judge_evidence(mock_client, "q?", [_chunk("t1")])
    assert result["verdict"] == "insufficient"  # failed closed after retry
    assert mock_client.messages.create.call_count == 2


def test_judge_evidence_missing_cross_study_flag_is_a_schema_failure():
    from src.rag.evidence_judge import judge_evidence

    bad = _valid_evidence_payload()
    del bad["cross_study_comparison"]
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [_mock_response(json.dumps(bad)), _mock_response(json.dumps(_valid_evidence_payload()))]
    result = judge_evidence(mock_client, "q?", [_chunk("t1")])
    assert result["cross_study_comparison"] is True
    assert mock_client.messages.create.call_count == 2


def test_judge_evidence_max_tokens_stop_is_treated_as_parse_failure():
    from src.rag.evidence_judge import judge_evidence

    truncated = _mock_response(json.dumps(_valid_evidence_payload()))
    truncated.stop_reason = "max_tokens"
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [truncated, truncated]
    result = judge_evidence(mock_client, "q?", [_chunk("t1")])
    assert result["verdict"] == "insufficient"


def test_audit_note_adds_cross_study_and_personal_dosing_lines():
    from src.rag.chain import _evidence_audit_note

    note = _evidence_audit_note({"population_mismatch": False, "cross_study_comparison": True}, {"personal_dosing": True})
    assert "separate studies" in note
    assert "Do not recommend any dose" in note
    assert _evidence_audit_note({"population_mismatch": False}, {"personal_dosing": False}) == ""


def test_judge_fields_surfaces_new_fields_and_tolerates_bypassed_results():
    from src.rag.chain import _judge_fields

    fields = _judge_fields(
        {"scope": "in_charter", "area": "diet_nutrition", "outcome": "HbA1c", "personal_dosing": True},
        {"on_topic": True, "cross_study_comparison": False},
    )
    assert fields == {"outcome": "HbA1c", "area": "diet_nutrition", "personal_dosing": True,
                      "on_topic": True, "cross_study_comparison": False}
    bypassed = _judge_fields({"scope": "in_charter", "topic": None, "reason": "forced"})
    assert bypassed["personal_dosing"] is False and bypassed["on_topic"] is None
