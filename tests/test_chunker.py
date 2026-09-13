from src.ingest.chunker import split_pubmed_documents_sectioned


def test_sectioned_chunking_keeps_short_sections_as_one_chunk_each():
    doc = {
        "pmid": "1", "title": "t", "journal": "j", "year": "2024",
        "publication_type": "Review", "population": "type2", "population_confidence": "high",
        "sections": [
            {"label": "BACKGROUND", "text": "Short background text here."},
            {"label": "METHODS", "text": "Short methods text here."},
            {"label": "RESULTS", "text": "Short results text here."},
            {"label": "CONCLUSIONS", "text": "Short conclusions text here."},
        ],
    }
    chunks = split_pubmed_documents_sectioned([doc], chunk_size_words=300, overlap_words=50)
    assert len(chunks) == 4
    assert [c["metadata"]["section"] for c in chunks] == ["BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS"]
    assert [c["metadata"]["chunk_index"] for c in chunks] == [0, 1, 2, 3]
    assert chunks[0]["text"] == "Short background text here."
    assert chunks[0]["metadata"]["population_confidence"] == "high"


def test_sectioned_chunking_never_crosses_a_section_boundary():
    long_methods = " ".join(f"word{i}" for i in range(400))
    doc = {
        "pmid": "2", "title": "t", "journal": "j", "year": "2024",
        "publication_type": "Review", "population": "type2", "population_confidence": "low",
        "sections": [
            {"label": "BACKGROUND", "text": "Short intro."},
            {"label": "METHODS", "text": long_methods},
        ],
    }
    chunks = split_pubmed_documents_sectioned([doc], chunk_size_words=300, overlap_words=50)
    # The long METHODS section splits into 2 pieces via the sliding window,
    # but neither piece ever contains the BACKGROUND text or vice versa.
    background_chunks = [c for c in chunks if c["metadata"]["section"] == "BACKGROUND"]
    methods_chunks = [c for c in chunks if c["metadata"]["section"] == "METHODS"]
    assert len(background_chunks) == 1
    assert len(methods_chunks) == 2
    assert "Short intro" not in methods_chunks[0]["text"]
    assert "word0" not in background_chunks[0]["text"]


def test_sectioned_chunking_falls_back_for_unstructured_abstracts():
    doc = {
        "pmid": "3", "title": "t", "journal": "j", "year": "2024",
        "publication_type": "Review", "population": "mixed", "population_confidence": "low",
        # No "sections" key at all — an old-style unstructured abstract.
        "abstract": "Plain unstructured abstract text with no section labels.",
    }
    chunks = split_pubmed_documents_sectioned([doc], chunk_size_words=300, overlap_words=50)
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["section"] == ""
    assert chunks[0]["text"] == "Plain unstructured abstract text with no section labels."
