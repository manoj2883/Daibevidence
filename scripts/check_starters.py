"""
Checks that every starter question on the landing page (src/api/index.html) gets status
answered or answered_partial from the v3 pipeline. Stops each question at the "sources"/"refusal"
event, since the status is decided before generation, so no answer is generated.

    python -m scripts.check_starters
    python -m scripts.check_starters --question "Another candidate?"
"""
import argparse
import html
import json
import re

from dotenv import load_dotenv

from src.rag.pipeline import stream_answer_v3

INDEX_PATH = "src/api/index.html"
OK_STATUSES = {"answered", "answered_partial"}


def starter_questions(path=INDEX_PATH):
    page = open(path, encoding="utf-8").read()
    empty_state = page[page.index('<div id="empty-state">'):page.index("</main>")]
    return [html.unescape(q) for q in re.findall(r'class="starter-card"[^>]*data-question="([^"]+)"', empty_state)]


def status_of(question):
    for ev in stream_answer_v3(question):
        if ev["event"] in ("sources", "refusal"):
            d = ev["data"]
            return d["status"], d.get("evidence_reason")
        if ev["event"] == "error":
            return "error", ev["data"].get("message")
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", action="append")
    args = parser.parse_args()
    load_dotenv()
    questions = args.question or starter_questions()
    results = []
    for q in questions:
        status, reason = status_of(q)
        results.append({"question": q, "status": status, "ok": status in OK_STATUSES, "evidence_reason": reason})
        print(f"{'OK ' if status in OK_STATUSES else 'BAD'} {status:18s} {q}", flush=True)
    print(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
