from unittest.mock import patch

import pytest

from src.rag.greetings import ABOUT_TEXT, BYE_TEXT, HELLO_TEXT, THANKS_TEXT, match_greeting


@pytest.mark.parametrize("text", ["hi", "Hi!", "  HELLO  ", "hey.", "Hey there!!", "Good morning", "good EVENING!", "hi there"])
def test_hello_variants(text):
    reply = match_greeting(text)
    assert reply == {"kind": "hello", "message": HELLO_TEXT, "show_starters": True}


@pytest.mark.parametrize("text,kind,message,chips", [
    ("thanks", "thanks", THANKS_TEXT, False),
    ("Thank you!", "thanks", THANKS_TEXT, False),
    ("bye", "bye", BYE_TEXT, False),
    ("Goodbye.", "bye", BYE_TEXT, False),
    ("Who are you?", "about", ABOUT_TEXT, True),
    ("what can you do", "about", ABOUT_TEXT, True),
])
def test_other_small_talk(text, kind, message, chips):
    assert match_greeting(text) == {"kind": kind, "message": message, "show_starters": chips}


@pytest.mark.parametrize("text", [
    "hi, does metformin cause weight gain?",
    "Thanks, but what about type 1 diabetes?",
    "What is HbA1c?",
    "high blood sugar in the morning",  # starts with "hi" but is a question
    "",
])
def test_real_questions_are_not_small_talk(text):
    assert match_greeting(text) is None


def test_greeting_skips_the_pipeline_entirely():
    from src.rag import pipeline

    with patch.object(pipeline, "stream_answer_v3") as v3, patch.object(pipeline, "offline_events") as offline:
        events = list(pipeline.run_pipeline("Hi!"))
    v3.assert_not_called()
    offline.assert_not_called()
    assert events[0] == {"event": "greeting", "data": {"status": "greeting", "kind": "hello", "message": HELLO_TEXT, "show_starters": True}}
    assert events[-1]["event"] == "done"


def test_greetings_work_in_offline_mode(monkeypatch):
    from src.rag import pipeline

    monkeypatch.setenv("OFFLINE_MODE", "1")
    with patch.object(pipeline, "offline_events") as offline:
        events = list(pipeline.run_pipeline("thanks"))
    offline.assert_not_called()
    assert events[0]["data"]["message"] == THANKS_TEXT


def test_questions_still_reach_the_pipeline():
    from src.rag import pipeline

    with patch.object(pipeline, "stream_answer_v3", return_value=iter([{"event": "done", "data": {}}])) as v3:
        list(pipeline.run_pipeline("What is HbA1c?"))
    v3.assert_called_once()
