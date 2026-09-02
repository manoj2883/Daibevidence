from src.ingest.local_embeddings import EMBEDDING_DIM, LocalSentenceTransformerEmbeddings


def test_embed_query_dimension():
    embeddings = LocalSentenceTransformerEmbeddings()
    vector = embeddings.embed_query("Diet and glycemic control in type 2 diabetes")
    assert len(vector) == EMBEDDING_DIM == 384


def test_embed_documents_batch():
    embeddings = LocalSentenceTransformerEmbeddings()
    vectors = embeddings.embed_documents([
        "Weight loss improves insulin sensitivity.",
        "Body composition changes with bariatric surgery.",
    ])
    assert len(vectors) == 2
    for vector in vectors:
        assert len(vector) == EMBEDDING_DIM


def test_embed_query_is_deterministic():
    embeddings = LocalSentenceTransformerEmbeddings()
    text = "Glycemic control and dietary intervention"
    assert embeddings.embed_query(text) == embeddings.embed_query(text)
