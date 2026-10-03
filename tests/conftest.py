import pytest


@pytest.fixture(autouse=True)
def _pass_through_sentence_verifier(request, monkeypatch):
    """
    Pipeline tests mock the Anthropic client, which can't answer the sentence verifier (it would
    fail closed to not_covered). Default it to "every sentence supported and answering the core
    claim"; tests of the verifier itself opt out with @pytest.mark.real_verifier.
    """
    if request.node.get_closest_marker("real_verifier"):
        return
    from src.rag import pipeline

    def fake_verify(client, question, claim, sentences, chunks):
        return {"labels": [{"label": "supported", "answers": "core"} for _ in sentences], "model": "fake", "error": None}

    monkeypatch.setattr(pipeline, "verify_sentences", fake_verify)


def pytest_configure(config):
    config.addinivalue_line("markers", "real_verifier: use the real sentence verifier code path")
