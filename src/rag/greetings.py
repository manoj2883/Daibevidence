"""
Small talk handled before the pipeline: no retrieval, no model call.

A message counts as small talk only if, after lowercasing, removing punctuation and collapsing
spaces, it is exactly one of the phrases below. "hi, does metformin cause weight gain?" is not
small talk: it goes to the pipeline like any other question.
"""
import re
from typing import Dict, Optional

HELLO_TEXT = "Hi! I answer diabetes questions using published research. What would you like to know?"
ABOUT_TEXT = (
    "I'm DiabEvidence. I answer questions about diabetes (type 1, type 2, gestational and prediabetes) "
    "using published research, and I show the studies behind every answer. I can't give personal medical "
    "advice. What would you like to know?"
)
THANKS_TEXT = "You're welcome!"
BYE_TEXT = "Take care!"

_PHRASES = {
    "hello": ["hi", "hii", "hiya", "hello", "hey", "hey there", "hi there", "hello there", "yo",
              "good morning", "good afternoon", "good evening", "morning", "evening"],
    "about": ["who are you", "what are you", "what can you do", "what do you do", "help"],
    "thanks": ["thanks", "thank you", "thank you so much", "thanks a lot", "thanks so much", "thx", "ty",
               "many thanks", "cheers"],
    "bye": ["bye", "goodbye", "good bye", "bye bye", "see you", "see ya", "good night"],
}
_REPLIES = {"hello": HELLO_TEXT, "about": ABOUT_TEXT, "thanks": THANKS_TEXT, "bye": BYE_TEXT}
# Kinds whose reply invites a question, so the UI shows the starter-question chips with it.
_SHOW_STARTERS = {"hello", "about"}
_LOOKUP = {phrase: kind for kind, phrases in _PHRASES.items() for phrase in phrases}


def normalize(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def match_greeting(text: str) -> Optional[Dict[str, object]]:
    """{"kind", "message", "show_starters"} for small talk, else None."""
    kind = _LOOKUP.get(normalize(text))
    if kind is None:
        return None
    return {"kind": kind, "message": _REPLIES[kind], "show_starters": kind in _SHOW_STARTERS}


def greeting_events(text: str):
    """The event stream for a greeting: one "greeting" event and a "done"."""
    reply = match_greeting(text)
    yield {"event": "greeting", "data": {"status": "greeting", **reply}}
    yield {"event": "done", "data": {"status": "greeting", "state": "greeting"}}
