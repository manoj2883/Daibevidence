import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.ingest.local_embeddings import LocalSentenceTransformerEmbeddings
from src.rag.chain import stream_answer
from src.rag.retriever import get_namespace, validate_namespace

# Load environment variables
load_dotenv()

logger = logging.getLogger("diabevidence")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the local embedding model once at boot instead of lazily on the
    # first query, so cold starts are predictable and config issues fail fast.
    LocalSentenceTransformerEmbeddings()

    # Fail fast if the configured Pinecone namespace doesn't exist or is
    # empty, rather than starting up and silently answering from the wrong
    # (or no) data — a query against an empty/nonexistent namespace doesn't
    # error, it just returns zero matches, indistinguishable at query time
    # from "no evidence for this question". Never falls back to another
    # namespace: validate_namespace() raises instead.
    namespace = get_namespace()
    index_name = os.environ.get("PINECONE_INDEX_NAME", "").strip()
    vector_count = validate_namespace(namespace)
    app.state.namespace = namespace
    app.state.namespace_vector_count = vector_count
    startup_msg = f"Pinecone namespace resolved: namespace={namespace!r} vector_count={vector_count} index={index_name!r}"
    logger.info(startup_msg)
    print(f"[startup] {startup_msg}")

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
    Service health check endpoint. namespace/namespace_vector_count reflect
    what was validated at startup (see lifespan) — reading them here lets a
    running deployment be checked without reading logs, per the "the app
    must query v2_5k, always" fix: the app couldn't have started at all if
    the resolved namespace were missing or empty.
    """
    return {
        "status": "healthy",
        "claude_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "pinecone_configured": bool(os.environ.get("PINECONE_API_KEY") and os.environ.get("PINECONE_INDEX_NAME")),
        "namespace": getattr(app.state, "namespace", None),
        "namespace_vector_count": getattr(app.state, "namespace_vector_count", None),
    }


@app.post("/query", tags=["RAG"])
async def query_rag(request: QueryRequest):
    """
    Submit a query to the RAG pipeline. Streams a server-sent-events response:

    State enum: answered / answered_low_confidence / no_evidence_for_claim /
    out_of_scope. Only out_of_scope is known before generation (nothing
    cleared the floor in either the evidence or background tier); the
    others depend on which chunks the model actually cites, so "sources"
    carries a provisional guess and "done" carries the authoritative state.

    - "sources" — retrieved chunks + population info + provisional state +
      score distribution + timing_ms.retrieval (sent once, before generation)
    - "contradictions" — conflicting findings detected across excerpts (an
      empty list if none) + timing_ms.contradiction_check, sent once,
      before any "sentence" events
    - "sentence" — one {sentence, chunk_ids, pmids, supported, source_type}
      object, sent as soon as it completes in the stream (citation
      attribution is structural, not inline text markup); source_type is
      "evidence"/"background"/None
    - "done"    — generation finished, carries the authoritative state, the
      disclaimer, the groundedness summary (overall + evidence-only +
      background-only, since a background-only answer looking "grounded"
      would hide that no study actually backs it), timing_ms (retrieval /
      generation / total), real token usage, and "truncated" (true if the
      response was cut off by the model's output token limit rather than
      reaching a natural end — a generation failure, never to be read as
      "no evidence exists")
    - "refusal" — out_of_scope: nothing cleared the floor in either tier.
      Closest scores found, what the system covers, and the score
      distribution. No Claude call made.
    - "error"   — misconfiguration or generation failure
    """
    return StreamingResponse(_sse_stream(request.question), media_type="text/event-stream")
