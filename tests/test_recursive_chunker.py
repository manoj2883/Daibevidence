from src.ingest.recursive_chunker import (
    chunk_document_text,
    is_retracted,
    make_header,
    split_background_documents_recursive,
    split_pubmed_documents_recursive,
    split_sentences,
    study_design,
)
from src.ingest.local_embeddings import count_tokens


def words(text):
    """Deterministic stand-in for the MiniLM token counter: one token per word."""
    return len(text.split())


HEADER = "T | 2024 | Type 2 diabetes | RCT"
H = len(HEADER.split())  # header size under the word counter


def test_short_document_is_one_chunk_with_all_sections():
    sections = [{"label": "BACKGROUND", "text": "One. Two."}, {"label": "RESULTS", "text": "Three."}]
    chunks = chunk_document_text(HEADER, sections, max_tokens=50, token_counter=words)
    assert len(chunks) == 1
    assert chunks[0]["body"] == "BACKGROUND: One. Two. RESULTS: Three."


def test_sections_that_fit_are_never_split():
    a = "Alpha beta gamma delta. Epsilon zeta eta theta."  # 8 words
    b = "Iota kappa lambda mu. Nu xi omicron pi."
    sections = [{"label": "", "text": a}, {"label": "", "text": b}]
    chunks = chunk_document_text(HEADER, sections, max_tokens=H + 8, token_counter=words)
    assert [c["body"] for c in chunks] == [a, b]


def test_never_cuts_mid_sentence_and_carries_last_sentence_forward():
    sentences = [f"Sentence {i} has five words." for i in range(6)]  # 5 words each
    sections = [{"label": "", "text": " ".join(sentences)}]
    chunks = chunk_document_text(HEADER, sections, max_tokens=H + 15, token_counter=words)
    for c in chunks:
        assert c["body"].endswith(".")
        assert words(f"{HEADER}\n{c['body']}") <= H + 15
    # Overlap: each chunk after the first starts with the previous chunk's last sentence.
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt["body"].startswith(split_sentences(prev["body"])[-1])
    joined = " ".join(c["body"] for c in chunks)
    assert all(s in joined for s in sentences)


def test_overlong_sentence_falls_back_to_words():
    long_sentence = " ".join(f"w{i}" for i in range(40)) + "."
    limit = H + 10
    chunks = chunk_document_text(HEADER, [{"label": "", "text": long_sentence}], max_tokens=limit, token_counter=words)
    assert len(chunks) == 4
    assert all(words(f"{HEADER}\n{c['body']}") <= limit for c in chunks)
    assert " ".join(c["body"] for c in chunks) == long_sentence


def test_sentence_splitter_keeps_abbreviations_and_decimals():
    text = "HbA1c fell by 0.5% vs. placebo, e.g. in adults. Weight also fell. (n = 40) Results held."
    assert split_sentences(text) == [
        "HbA1c fell by 0.5% vs. placebo, e.g. in adults.",
        "Weight also fell.",
        "(n = 40) Results held.",
    ]


def test_study_design_badges():
    assert study_design("Journal Article; Systematic Review; Meta-Analysis") == "Meta-analysis"
    assert study_design("Randomized Controlled Trial; Journal Article") == "RCT"
    assert study_design("Journal Article; Review") == "Review"
    assert study_design("Journal Article; Observational Study") == "Observational"
    assert study_design("Clinical Trial, Phase III; Journal Article") == "Clinical trial"


def test_retracted_abstracts_are_excluded_and_metadata_is_stored():
    docs = [
        {"pmid": "1", "title": "Kept", "year": "2024", "population": "type2",
         "publication_type": "Journal Article; Randomized Controlled Trial",
         "sections": [{"label": "RESULTS", "text": "Glucose fell."}]},
        {"pmid": "2", "title": "Gone", "year": "2024", "population": "type2",
         "publication_type": "Journal Article; Retracted Publication",
         "sections": [{"label": "", "text": "Retracted text."}]},
    ]
    assert is_retracted(docs[1]["publication_type"])
    chunks = split_pubmed_documents_recursive(docs)
    assert len(chunks) == 1
    meta = chunks[0]["metadata"]
    assert meta["parent_id"] == "1" and meta["pmid"] == "1"
    assert meta["publication_type"] == "Journal Article; Randomized Controlled Trial"
    assert meta["study_design"] == "RCT"
    assert chunks[0]["text"].startswith("Kept | 2024 | Type 2 diabetes | RCT\n")


def test_real_token_limit_includes_header():
    long_text = " ".join(f"Patients in group {i} lowered their fasting glucose levels substantially." for i in range(60))
    docs = [{"pmid": "9", "title": "A long trial title about glycemic control", "year": "2023",
             "population": "mixed", "publication_type": "Review", "abstract": long_text}]
    chunks = split_pubmed_documents_recursive(docs)
    assert len(chunks) > 1
    assert all(count_tokens(c["text"]) <= 250 for c in chunks)


def test_background_pages_get_background_header_and_parent():
    docs = [{"publisher": "ADA", "title": "Prediabetes", "source_url": "u", "population": "prediabetes", "text": "Prediabetes means glucose is high."}]
    chunks = split_background_documents_recursive(docs)
    assert chunks[0]["text"].startswith(make_header("Prediabetes", "", "prediabetes", "Background (ADA)"))
    assert chunks[0]["metadata"]["source_type"] == "background"
    assert chunks[0]["metadata"]["parent_id"] == "bg_ada_prediabetes"
