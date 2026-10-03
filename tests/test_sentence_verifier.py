import json
from unittest.mock import MagicMock, patch

import pytest

from src.rag.sentence_verifier import core_claim, decide, verify_sentences
from src.rag.types import RetrievedChunk


def s(text, ids=(1,)):
    return {"sentence": text, "chunk_ids": list(ids)}


def L(label, answers="core"):
    return {"label": label, "answers": answers}


def test_all_supported_is_ok_and_status_unchanged():
    out = decide("answered", [s("a"), s("b")], [L("supported"), L("non_factual", "none")])
    assert out["outcome"] == "ok" and out["status"] == "answered" and out["keep"] == [0, 1]


def test_unsupported_sentence_removed_and_status_only_goes_down():
    out = decide("answered", [s("a"), s("b")], [L("supported"), L("unsupported", "part")])
    assert out["status"] == "answered_partial" and out["keep"] == [0] and out["removed"] == ["b"]
    out = decide("answered_partial", [s("a"), s("b")], [L("supported"), L("unsupported", "part")])
    assert out["status"] == "answered_partial"


def test_no_surviving_sentence_on_the_core_claim_is_not_covered():
    out = decide("answered", [s("a"), s("b")], [L("unsupported"), L("supported", "part")])
    assert out["outcome"] == "not_covered" and out["reason"] == "no supported sentence answers the core claim"


def test_id_35_shape_is_not_covered():
    """One supported sentence that doesn't answer the question, then a gap statement (v3 id 35)."""
    sentences = [
        s("Peripheral arterial disease is described as one of the factors that can weaken the feet.", [6]),
        s("This information comes from research covering a mix of diabetes types.", [6]),
        s("The studies gathered here don't explain how peripheral arterial disease is checked during a foot exam.", []),
    ]
    out = decide("answered_partial", sentences, [L("supported", "none"), L("non_factual", "none"), L("non_factual", "none")])
    assert out["outcome"] == "not_covered"


def test_verifier_failure_fails_closed():
    assert decide("answered", [s("a")], None)["outcome"] == "not_covered"


def test_core_claim_comes_from_the_evidence_judge():
    ev = {"propositions": [{"claim": "x", "core": False}, {"claim": "fiber lowers HbA1c", "core": True}]}
    assert core_claim("q?", ev) == "fiber lowers HbA1c"
    assert core_claim("q?", {"propositions": []}) == "q?"


def _response(text):
    block = MagicMock(type="text", text=text)
    return MagicMock(content=[block], stop_reason="end_turn", model="claude-haiku-4-5-20251001")


@pytest.mark.real_verifier
def test_verify_sentences_uses_haiku_and_only_cited_passages():
    from src.rag.chain import get_model
    from src.rag.judge import get_judge_model

    client = MagicMock()
    client.messages.create.return_value = _response(json.dumps(
        {"sentences": [{"index": 1, "label": "supported", "answers": "core"}]}))
    chunks = [RetrievedChunk(text="PASSAGE ONE"), RetrievedChunk(text="PASSAGE TWO")]
    out = verify_sentences(client, "q?", "claim", [s("a", [2])], chunks)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == get_judge_model() != get_model()  # verifier Haiku, grader stays Sonnet
    assert "PASSAGE TWO" in kwargs["system"] and "PASSAGE ONE" not in kwargs["system"]
    assert out["labels"] == [{"label": "supported", "answers": "core"}]


@pytest.mark.real_verifier
def test_verify_sentences_retries_then_fails_closed():
    client = MagicMock()
    client.messages.create.return_value = _response('{"sentences": [{"index": 1, "label": "maybe"}]}')
    out = verify_sentences(client, "q?", "claim", [s("a")], [RetrievedChunk(text="p")])
    assert out["labels"] is None and client.messages.create.call_count == 2


@pytest.mark.real_verifier
def test_pipeline_removes_unsupported_and_refuses_when_core_is_lost(tmp_path):
    from src.rag import pipeline
    from tests.test_answer_checks import _run

    def verify_one_bad(client, question, claim, sentences, chunks):
        return {"labels": [L("supported"), L("unsupported", "part")][:len(sentences)], "model": "fake", "error": None}

    with patch.object(pipeline, "verify_sentences", side_effect=verify_one_bad):
        events = _run([s("Fiber helps."), s("Oats cure diabetes.")], tmp_path)
    assert [e["data"]["sentence"] for e in events if e["event"] == "sentence"] == ["Fiber helps."]
    assert [e for e in events if e["event"] == "done"][0]["data"]["status"] == "answered_partial"
    log = [json.loads(l) for l in (tmp_path / "checks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log[-1]["stage"] == "verifier" and log[-1]["removed"] == ["Oats cure diabetes."]

    def verify_core_bad(client, question, claim, sentences, chunks):
        return {"labels": [L("unsupported") for _ in sentences], "model": "fake", "error": None}

    with patch.object(pipeline, "verify_sentences", side_effect=verify_core_bad):
        events = _run([s("Fiber helps.")], tmp_path)
    assert [e for e in events if e["event"] == "refusal"][0]["data"]["status"] == "not_covered"
