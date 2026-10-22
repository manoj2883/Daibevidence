"""Unit tests for the Ragas dataset builder and grade mapping (scripts.run_ragas_eval). No API calls."""
import pytest

from scripts.run_ragas_eval import build_dataset, metrics_for, ragas_label


@pytest.fixture(scope="module")
def rows():
    return {r["id"]: r for r in build_dataset()}


def test_dataset_covers_the_43_dev_questions_and_never_the_held_out_ids(rows):
    assert sorted(rows) == list(range(1, 44))


def test_answered_rows_use_the_generators_parents_with_full_text(rows):
    r = rows[3]  # answered_partial in the verifier run: 2 of 6 parents passed to generation
    assert r["status"] == "answered_partial" and r["contexts_kind"] == "used_for_generation"
    assert r["context_ids"] == ["36287562", "36297075"]
    assert all(len(c.split()) > 100 for c in r["retrieved_contexts"])


def test_not_covered_rows_carry_retrieved_parents_and_no_response(rows):
    r = rows[6]
    assert r["status"] == "not_covered" and r["response"] is None
    assert r["contexts_kind"] == "retrieved_not_used" and len(r["retrieved_contexts"]) == 6


def test_metric_selection(rows):
    assert metrics_for(rows[1]) == ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    assert metrics_for(rows[6]) == ["context_precision", "context_recall"]  # in scope, not answered
    assert metrics_for(rows[23]) == []  # in scope but no reference answer, not answered
    assert metrics_for(rows[25]) == ["faithfulness", "answer_relevancy"]  # out of scope, answered
    assert metrics_for(rows[43]) == []


@pytest.mark.parametrize("score,label", [(1.0, "SUPPORTED"), (0.95, "PARTIALLY_SUPPORTED"),
                                         (0.5, "PARTIALLY_SUPPORTED"), (0.49, "UNSUPPORTED"), (0.0, "UNSUPPORTED")])
def test_ragas_label_mapping(score, label):
    assert ragas_label(score) == label
