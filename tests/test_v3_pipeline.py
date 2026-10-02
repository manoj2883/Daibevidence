from unittest.mock import MagicMock, patch

from src.rag.hybrid_retriever import dedupe_by_parent, order_passages, reciprocal_rank_fusion, tokenize
from src.rag.query_rewriter import validate
from src.rag.safety import MEDICATION_NOTE, names_unmentioned_drugs, touches_medication
from src.rag.types import RetrievedChunk


# --- retrieval ----------------------------------------------------------------------------------

def test_rrf_rewards_items_found_by_both_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "d"]], k=60)
    assert fused["c"] == 1 / 63 + 1 / 61
    assert max(fused, key=fused.get) == "c"
    assert set(fused) == {"a", "b", "c", "d"}


def test_dedupe_keeps_best_chunk_per_parent_and_limits():
    fused = {"p1_0": 0.5, "p1_3": 0.9, "p2_0": 0.7, "p3_1": 0.1}
    parent_of = {"p1_0": "p1", "p1_3": "p1", "p2_0": "p2", "p3_1": "p3"}
    assert dedupe_by_parent(fused, parent_of, limit=2) == [("p1", "p1_3", 0.9), ("p2", "p2_0", 0.7)]


def test_order_passages_strongest_first_second_last():
    assert order_passages([1, 2, 3, 4, 5, 6]) == [1, 3, 4, 5, 6, 2]
    assert order_passages([1, 2]) == [1, 2]


def test_bm25_tokenizer_keeps_hyphenated_and_numeric_terms():
    assert tokenize("Does GLP-1 lower HbA1c by 0.5%?") == ["glp-1", "lower", "hba1c", "0.5"]


# --- query step ---------------------------------------------------------------------------------

def test_query_step_validation_caps_and_dedupes_keywords():
    out = validate({"search_query": " morning hyperglycemia ", "keywords": ["a", "A", "b", "c", "d", "e", "f"], "personal_dosing": False})
    assert out == {"search_query": "morning hyperglycemia", "keywords": ["a", "b", "c", "d", "e"], "personal_dosing": False}


def test_query_step_validation_rejects_missing_dosing_flag():
    assert validate({"search_query": "x", "keywords": []}) is None
    assert validate({"search_query": "", "keywords": [], "personal_dosing": False}) is None


# --- safety -------------------------------------------------------------------------------------

def test_unmentioned_drug_detection():
    q = "Does a low-carb diet help type 2 diabetes?"
    assert names_unmentioned_drugs(q, ["Some diabetes medicines may need adjusting."]) == []
    assert names_unmentioned_drugs(q, ["People taking metformin or SGLT2 inhibitors saw benefits."]) == ["sglt2 inhibitors", "metformin"]
    assert names_unmentioned_drugs("Does metformin interact with food?", ["Metformin is taken with meals."]) == []


def test_touches_medication():
    assert touches_medication("Does cinnamon lower blood sugar?", ["It lowered glucose a little."])
    assert touches_medication("Does walking help?", ["Some diabetes medicines also lower glucose."])
    assert not touches_medication("Does walking help?", ["Walking lowered blood sugar after meals."])


# --- pipeline routing (mocked) -------------------------------------------------------------------

def _chunks():
    return [RetrievedChunk(text="Full abstract.", metadata={"source_type": "evidence", "pmid": "1", "population": "type2",
                                                            "study_design": "RCT", "parent_id": "1"}, score=0.03)]


def _run(query, verdict="sufficient", audience="patient", stream_text=None):
    from src.rag import pipeline

    evidence = {"propositions": [{"claim": "c", "core": True, "direct": [1], "partial": [], "absent": False}],
                "question_population": "type 2", "evidence_populations": ["type2"], "population_mismatch": False,
                "on_topic": True, "cross_study_comparison": False, "verdict": verdict, "reason": "r", "model": "m"}

    def fake_generation(client, system_prompt, question, chunks, t, out):
        fake_generation.prompt = system_prompt
        fake_generation.question = question
        out.update(sentences=[{"sentence": stream_text or "It helps.", "chunk_ids": [1], "supported": True, "new_paragraph": False}],
                   usage=None, truncated=False, model="gen")
        yield {"event": "sentence", "data": {"sentence": stream_text or "It helps."}}

    with patch.object(pipeline, "get_client", return_value=MagicMock()), \
         patch.object(pipeline, "rewrite_query", return_value=query), \
         patch.object(pipeline, "hybrid_retrieve", return_value=(_chunks(), {"dense": [], "bm25": [], "fused": []})) as hr, \
         patch.object(pipeline, "judge_evidence", return_value=evidence) as je, \
         patch.object(pipeline, "stream_generation", side_effect=fake_generation), \
         patch.object(pipeline, "log_query_event"):
        events = list(pipeline.stream_answer_v3("high sugar in the morning?", audience=audience))
    return events, hr, je, fake_generation


QUERY = {"search_query": "morning hyperglycemia", "keywords": ["dawn phenomenon"], "personal_dosing": False}


def test_personal_dosing_routes_to_see_clinician_without_retrieval():
    events, hr, je, _ = _run({**QUERY, "personal_dosing": True})
    refusal = [e for e in events if e["event"] == "refusal"][0]["data"]
    assert refusal["status"] == "see_clinician"
    hr.assert_not_called()
    je.assert_not_called()


def test_insufficient_is_not_covered_with_fixed_message():
    events, _, _, _ = _run(QUERY, verdict="insufficient")
    refusal = [e for e in events if e["event"] == "refusal"][0]["data"]
    assert refusal["status"] == "not_covered"
    assert refusal["message"] == "Our research library doesn't cover this question yet."


def test_query_step_output_never_reaches_judge_or_generation():
    events, hr, je, gen = _run(QUERY)
    assert hr.call_args.args[0] == "morning hyperglycemia"  # retrieval uses the rewrite
    assert je.call_args.args[1] == "high sugar in the morning?"  # judge sees the original question
    assert gen.question == "high sugar in the morning?"
    assert "morning hyperglycemia" not in gen.prompt and "dawn phenomenon" not in gen.prompt
    sentences = [e["data"]["sentence"] for e in events if e["event"] == "sentence"]
    assert all("hyperglycemia" not in s for s in sentences)
    done = [e for e in events if e["event"] == "done"][0]["data"]
    assert done["status"] == "answered"


def test_patient_medication_answer_ends_with_note_and_clinician_does_not():
    events, _, _, gen = _run(QUERY, stream_text="Some diabetes medicines lower glucose.")
    sentences = [e["data"]["sentence"] for e in events if e["event"] == "sentence"]
    assert sentences[-1] == MEDICATION_NOTE
    assert "NEVER name a specific drug" in gen.prompt

    events, _, _, gen = _run(QUERY, audience="clinician", stream_text="Metformin lowers glucose.")
    sentences = [e["data"]["sentence"] for e in events if e["event"] == "sentence"]
    assert MEDICATION_NOTE not in sentences
    assert "PMID" in gen.prompt and "NEVER name a specific drug" not in gen.prompt


def test_scope_judge_flag_selects_two_stage_pipeline(monkeypatch):
    from src.rag import pipeline

    monkeypatch.setenv("SCOPE_JUDGE_ENABLED", "true")
    with patch.object(pipeline, "stream_answer_two_stage", return_value=iter(())) as legacy:
        pipeline.run_pipeline("q")
    legacy.assert_called_once()
    monkeypatch.setenv("SCOPE_JUDGE_ENABLED", "false")
    assert not pipeline.scope_judge_enabled()
