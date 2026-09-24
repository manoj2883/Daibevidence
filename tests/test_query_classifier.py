from src.rag.query_classifier import (
    classify_and_decompose,
    classify_question_type,
    decompose_compound,
    overall_question_type,
)


def test_classify_definitional():
    assert classify_question_type("What is gestational diabetes?") == "definitional"


def test_classify_evidence_seeking():
    assert classify_question_type(
        "Does adherence to a Mediterranean-style diet reduce the risk of type 2 diabetes?"
    ) == "evidence_seeking"


def test_classify_defaults_to_evidence_seeking_when_ambiguous():
    assert classify_question_type("Metformin and lactic acidosis") == "evidence_seeking"


def test_decompose_does_not_split_compound_subject_bug_regression():
    # The actual bug: "type 1 and type 2 diabetes" is one question about two
    # entities, not two questions — "type 2 diabetes" doesn't start with a
    # question lead word, so it must NOT be split off as a separate part.
    assert decompose_compound("What is type 1 and type 2 diabetes") == ["What is type 1 and type 2 diabetes"]


def test_decompose_splits_genuine_compound_question():
    q = "What is diabetes and does metformin help with weight loss?"
    assert decompose_compound(q) == ["What is diabetes", "does metformin help with weight loss?"]


def test_decompose_splits_three_way():
    q = "What are the symptoms of prediabetes and how effective is metformin at preventing it?"
    assert decompose_compound(q) == [
        "What are the symptoms of prediabetes",
        "how effective is metformin at preventing it?",
    ]


def test_decompose_single_question_unchanged():
    q = "Does bariatric surgery lead to remission of type 2 diabetes?"
    assert decompose_compound(q) == [q]


def test_classify_and_decompose_orders_definitional_first():
    q = "Does metformin help with weight loss and what is prediabetes?"
    classified = classify_and_decompose(q)
    assert [c["type"] for c in classified] == ["definitional", "evidence_seeking"]
    assert classified[0]["question"] == "what is prediabetes?"


def test_overall_question_type_single():
    classified = classify_and_decompose("What is gestational diabetes?")
    assert overall_question_type(classified) == "definitional"


def test_overall_question_type_compound():
    classified = classify_and_decompose("What is diabetes and does metformin help with weight loss?")
    assert overall_question_type(classified) == "compound"
