import os
import pytest
from src.ingest.chunker import split_text, split_documents

def test_split_text():
    text = "This is a very long text. " * 50
    chunks = split_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 100

def test_split_documents():
    docs = [
        {"text": "Hello world. This is test document number one.", "metadata": {"source": "doc1"}},
        {"text": "Python is a programming language. It is easy to learn.", "metadata": {"source": "doc2"}}
    ]
    chunks = split_documents(docs, chunk_size=30, chunk_overlap=5)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert "text" in chunk
        assert "metadata" in chunk
        assert "source" in chunk["metadata"]
        assert "chunk_index" in chunk["metadata"]

@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"),
    reason="Gemini API key not configured in environment"
)
def test_embeddings_generation():
    from src.ingest.uploader import get_embeddings_model
    embeddings = get_embeddings_model()
    assert embeddings.model == "gemini-embedding-001"
    vector = embeddings.embed_query("Test embedding generation")
    assert len(vector) == 768

@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"),
    reason="Gemini API key not configured in environment"
)
def test_dual_model_rag_routing():
    from src.rag.chain import get_rag_chain
    rag_pipeline = get_rag_chain()
    
    # Test conversational query routing (bypasses Pinecone)
    response = rag_pipeline("Hello there! How are you?")
    assert "answer" in response
    assert response["source_documents"] == []

