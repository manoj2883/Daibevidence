import json
from unittest.mock import MagicMock, patch

from src.rag.answer_checks import check_answer, is_evidence_meta
from src.rag.types import RetrievedChunk


def s(text, ids):
    return {"sentence": text, "chunk_ids": ids, "supported": bool(ids), "new_paragraph": False}


# Uncited sentences taken from the v3 Run B answers (eval/results/v3_2026-10-02/eval_v3_runB_drugcheck.json).
META_FROM_RUN_B = [
    "The studies gathered here don't specifically focus on adults already flagged as high-risk for type 2 diabetes.",
    "It's worth knowing that this comparison is pieced together from separate studies that each tested their own diets.",
    "Overall, this evidence covers people with type 2 diabetes, prediabetes, and a general discussion of type 1 diabetes.",
    "This topic has been studied mainly in a mix of people with diabetes generally.",
    "These findings come from two separate studies rather than one study that directly compared them.",
    "The studies found here don't address how peripheral arterial disease is assessed during a diabetic foot exam.",
    "This information applies to a mix of diabetes types, including type 1 and type 2 diabetes.",
    "The evidence described below mostly comes from mixed groups of people with diabetic foot ulcers.",
    "The studies reviewed here do not describe what specific features or styles of footwear were used.",
]


def test_sentences_about_the_evidence_are_not_factual_claims():
    assert all(is_evidence_meta(t) for t in META_FROM_RUN_B)
    assert not is_evidence_meta("In gestational diabetes, which is diabetes that starts during pregnancy, the research results are mixed.")
    assert not is_evidence_meta("Studies found that walking lowers blood sugar.")


def test_clean_answer_is_ok():
    out = check_answer([s("Fiber helps.", [1]), s(META_FROM_RUN_B[0], [])], 2, ["11"])
    assert out["outcome"] == "ok" and out["keep"] == [0, 1]


def test_uncited_factual_sentence_is_removed_and_answer_becomes_partial():
    out = check_answer([s("Fiber helps.", [1]), s("In gestational diabetes the results are mixed.", [])], 1, ["11"])
    assert out["outcome"] == "partial"
    assert out["keep"] == [0]
    assert out["removed"] == ["In gestational diabetes the results are mixed."]


def test_zero_citations_is_not_covered():
    out = check_answer([s("Fiber helps.", []), s(META_FROM_RUN_B[5], [])], 3, ["11"])
    assert out["outcome"] == "not_covered" and out["reason"] == "no sentence cites a retrieved passage"


def test_citation_outside_retrieved_passages_is_not_covered():
    out = check_answer([s("Fiber helps.", [1]), s("Oats help.", [7])], 2, ["11", "22"])
    assert out["outcome"] == "not_covered" and out["bad_citations"] == [{"sentence_index": 1, "chunk_id": 7}]


def test_pmid_in_text_must_be_a_retrieved_paper():
    ok = check_answer([s("A 2021 meta-analysis (PMID 12345678) found a benefit.", [1])], 1, ["12345678"])
    assert ok["outcome"] == "ok"
    bad = check_answer([s("A 2021 meta-analysis (PMID 99999999) found a benefit.", [1])], 1, ["12345678"])
    assert bad["outcome"] == "not_covered" and bad["bad_citations"] == [{"sentence_index": 0, "pmid": "99999999"}]


# --- through the pipeline -------------------------------------------------------------------------

def _run(sentence_records, tmp_path, audience="clinician"):
    from src.rag import pipeline

    chunks = [RetrievedChunk(text="Full abstract.", metadata={"source_type": "evidence", "pmid": "11", "population": "type2",
                                                                "study_design": "RCT", "parent_id": "11"}, score=0.03)]
    evidence = {"propositions": [{"claim": "c", "core": True, "direct": [1], "partial": [], "absent": False}],
                "question_population": "type 2", "evidence_populations": ["type2"], "population_mismatch": False,
                "on_topic": True, "cross_study_comparison": False, "verdict": "sufficient", "reason": "r", "model": "m"}

    def fake_generation(client, system_prompt, question, chunks_, t, out):
        out.update(sentences=[dict(r) for r in sentence_records], usage=None, truncated=False, model="gen")
        yield {"event": "contradictions", "data": {"contradictions": []}}
        for r in sentence_records:
            yield {"event": "sentence", "data": dict(r)}

    with patch.object(pipeline, "get_client", return_value=MagicMock()), \
         patch.object(pipeline, "rewrite_query", return_value={"search_query": "q", "keywords": [], "personal_dosing": False}), \
         patch.object(pipeline, "hybrid_retrieve", return_value=(chunks, {"dense": [], "bm25": [], "fused": []})), \
         patch.object(pipeline, "judge_evidence", return_value=evidence), \
         patch.object(pipeline, "stream_generation", side_effect=fake_generation), \
         patch.object(pipeline, "log_query_event"), \
         patch.object(pipeline, "ANSWER_CHECK_LOG_PATH", str(tmp_path / "checks.jsonl")):
        return list(pipeline.stream_answer_v3("Does fiber help?", audience=audience))


def test_pipeline_drops_uncited_claim_and_reports_partial(tmp_path):
    events = _run([s("Fiber helps.", [1]), s("Oats cure diabetes.", [])], tmp_path)
    assert [e["data"]["sentence"] for e in events if e["event"] == "sentence"] == ["Fiber helps."]
    assert [e for e in events if e["event"] == "sources"][0]["data"]["status"] == "answered_partial"
    assert [e for e in events if e["event"] == "done"][0]["data"]["status"] == "answered_partial"
    assert "Oats cure diabetes." in (tmp_path / "checks.jsonl").read_text(encoding="utf-8")


def test_pipeline_refuses_uncited_answer(tmp_path):
    events = _run([s("Fiber helps.", [])], tmp_path)
    assert [e["event"] for e in events if e["event"] in ("sources", "sentence", "done")] == []
    assert [e for e in events if e["event"] == "refusal"][0]["data"]["status"] == "not_covered"


def test_pipeline_refuses_and_logs_bug_for_unknown_citation(tmp_path):
    events = _run([s("Fiber helps.", [1]), s("Oats help.", [4])], tmp_path)
    assert [e for e in events if e["event"] == "refusal"][0]["data"]["status"] == "not_covered"
    record = json.loads((tmp_path / "checks.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["bug"] is True and record["bad_citations"] == [{"sentence_index": 1, "chunk_id": 4}]
