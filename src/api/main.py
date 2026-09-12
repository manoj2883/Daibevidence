import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.ingest.local_embeddings import LocalSentenceTransformerEmbeddings
from src.rag.chain import stream_answer

# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the local embedding model once at boot instead of lazily on the
    # first query, so cold starts are predictable and config issues fail fast.
    LocalSentenceTransformerEmbeddings()
    yield


app = FastAPI(
    title="Diabevidence",
    description=(
        "A trustworthy retrieval-augmented QA system that answers diabetes questions "
        "using only published PubMed research, with visible citations and honest refusals. "
        "Direct Pinecone and Anthropic API calls only — no LangChain."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# CORS Middleware config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., description="The query/question to answer.")


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_stream(question: str):
    for item in stream_answer(question):
        yield _sse_event(item["event"], item["data"])


# Endpoints
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """
    Serve the single-page chat frontend.
    """
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h1>Frontend Interface Not Found</h1>", status_code=404)


@app.get("/health", tags=["Health"])
async def health():
    """
    Service health check endpoint.
    """
    return {
        "status": "healthy",
        "claude_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "pinecone_configured": bool(os.environ.get("PINECONE_API_KEY") and os.environ.get("PINECONE_INDEX_NAME"))
    }


@app.post("/query", tags=["RAG"])
async def query_rag(request: QueryRequest):
    """
    Submit a query to the RAG pipeline. Streams a server-sent-events response:

    - "sources" — retrieved chunks + population info + retrieval state
      ("answered" or "answered_low_confidence") + score distribution
      (sent once, before generation)
    - "contradictions" — conflicting findings detected across excerpts (an
      empty list if none), sent once, before any "token" events
    - "token"   — one text delta of the streaming answer
    - "done"    — generation finished, carries the disclaimer
    - "refusal" — nothing cleared the similarity floor: closest scores found,
      what the system covers, and the score distribution. No Claude call made.
    - "error"   — misconfiguration or generation failure
    """
    return StreamingResponse(_sse_stream(request.question), media_type="text/event-stream")
