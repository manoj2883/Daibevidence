import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

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
        task_type="retrieval_query",
        google_api_key=gemini_key,
        output_dimensionality=768
    )

def get_retriever(k: int = 4):
    """
    Initialize the Pinecone vector store and return a retriever.
    """
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY must be set in environment variables.")
    pinecone_key = pinecone_key.strip()
        
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME must be set in environment variables.")
    index_name = index_name.strip()

    embeddings = get_embeddings_model()
    
    # Initialize the Vector Store
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=pinecone_key
    )
    
    # Return as retriever
    return vector_store.as_retriever(search_kwargs={"k": k})
