import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load environment variables
load_dotenv()

def get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    """
    Initialize the Google Generative AI embeddings model.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set in environment variables.")
    gemini_key = gemini_key.strip()
    
    # Using gemini-embedding-001 with output_dimensionality=768
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        task_type="retrieval_document",
        google_api_key=gemini_key,
        output_dimensionality=768
    )

def upload_chunks(chunks: List[Dict[str, Any]], index_name: str = None) -> PineconeVectorStore:
    """
    Embed and upload chunks to Pinecone.
    Each chunk in the list should be a dictionary with 'text' and optional 'metadata'.
    """
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY must be set in environment variables.")
    pinecone_key = pinecone_key.strip()
        
    index_name = index_name or os.environ.get("PINECONE_INDEX_NAME")
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME must be set in environment variables.")
    index_name = index_name.strip()

    # Initialize Pinecone and check if index exists
    pc = Pinecone(api_key=pinecone_key)
    active_indexes = [index.name for index in pc.list_indexes()]
    if index_name not in active_indexes:
        raise ValueError(
            f"Index '{index_name}' does not exist in your Pinecone project. "
            f"Please create it first (dimension=768, metric=cosine)."
        )

    embeddings = get_embeddings_model()
    
    texts = [c["text"] for c in chunks]
    metadatas = [c.get("metadata", {}) for c in chunks]
    
    # Store in Pinecone
    vector_store = PineconeVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        index_name=index_name,
        metadatas=metadatas,
        pinecone_api_key=pinecone_key
    )
    
    return vector_store
