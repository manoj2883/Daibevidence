import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.ingest.config import NAMESPACE_V1_300, NAMESPACE_V2_5K
from src.ingest.local_embeddings import LocalSentenceTransformerEmbeddings
from src.rag.chain import stream_answer
from src.rag.retriever import get_namespace, validate_namespace

# Namespaces a client is allowed to request via /query's optional
# `namespace` field (comparison mode) — never pass an arbitrary
# client-supplied string straight to Pinecone. get_namespace()'s resolved
# production namespace is always allowed too, even if it's neither of
# these (e.g. a future namespace set only via PINECONE_NAMESPACE).
ALLOWED_QUERY_NAMESPACES = {NAMESPACE_V1_300, NAMESPACE_V2_5K}

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
    namespace: Optional[str] = Field(
        None,
        description=(
            "Optional Pinecone namespace override for side-by-side comparison "
            f"(e.g. running the same question against both {NAMESPACE_V1_300!r} "
            f"and {NAMESPACE_V2_5K!r}). Must be one of {sorted(ALLOWED_QUERY_NAMESPACES)!r} "
            "— any other value is rejected (400), never silently substituted or "
            "passed through to Pinecone as-is. Omit to use the server's configured "
            "production namespace."
        ),
    )


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_stream(question: str, namespace: Optional[str] = None):
    for item in stream_answer(question, namespace=namespace):
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
        # What /query's optional `namespace` field (comparison mode) will
        # accept — single source of truth, so the frontend never hardcodes
        # a duplicate of ALLOWED_QUERY_NAMESPACES that could drift from it.
        "available_namespaces": sorted(ALLOWED_QUERY_NAMESPACES),
    }


@app.post("/query", tags=["RAG"])
async def query_rag(request: QueryRequest):
    """
    Submit a query to the RAG pipeline. Streams a server-sent-events response.

    Two-stage answerability judge (src.rag.chain.stream_answer — see its
    docstring for the full routing table): Stage A decides scope
    (in_charter/adjacent/unrelated) from the question alone; Stage B, run
    only when Stage A doesn't already reject the question, audits the
    retrieved chunks per-proposition (question + chunks). `status` is the
    primary field going forward — `answered`, `answered_partial` (evidence
    only partially covers the question; generation is restricted to the
    directly-relevant chunks and names the gap), `in_scope_no_evidence`
    (a valid, in-charter question the corpus has no evidence for — never
    called "out of scope"), `answered_adjacent` (evidence is sufficient but
    the question falls outside the four core areas), or `out_of_scope`
    (unrelated, or adjacent with insufficient/partial evidence). `state`
    mirrors `status` for older consumers. Every event also carries `scope`,
    `scope_reason`, `evidence_verdict`, `evidence_reason`, `propositions`,
    `population_mismatch`, `question_population`, `evidence_populations`,
    and per-stage `timing_ms` (`scope`/`retrieval`/`evidence`/`generation`/
    `total`). The similarity floor no longer gates anything — its scores
    are still returned in `score_distribution` for the retrieval inspector
    only (see RECON.md).

    - "sources" — chunks actually used for generation + population info
      (requested_population is a list) + question_type + sub_questions +
      the scope/evidence audit fields above + score distribution (sent
      once, before generation — answered/answered_partial/answered_adjacent
      only)
    - "contradictions" — conflicting findings detected across excerpts (an
      empty list if none), sent once, before any "sentence" events
    - "sentence" — one {sentence, chunk_ids, pmids, supported, source_type,
      new_paragraph} object, sent as soon as it completes in the stream
    - "done" — generation finished; status is unchanged from the
      pre-generation decision (no post-hoc re-classification), plus the
      disclaimer, groundedness summary, per-stage timing, real token
      usage, and "truncated"
    - "refusal" — in_scope_no_evidence or out_of_scope. Message, what the
      system covers, the scope/evidence audit fields, score distribution.
      No generation call is made for either.
    - "error" — misconfiguration or generation failure

    `request.namespace` is optional and, if given, must be one of
    ALLOWED_QUERY_NAMESPACES (400 otherwise) — lets a client run the same
    question against v1_300 and v2_5k for a side-by-side comparison,
    without accepting an arbitrary namespace string from the client.
    Omitted, it falls through to the server's configured production
    namespace (unchanged behavior).
    """
    if request.namespace is not None and request.namespace not in ALLOWED_QUERY_NAMESPACES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid namespace {request.namespace!r}. Must be one of {sorted(ALLOWED_QUERY_NAMESPACES)}.",
        )
    return StreamingResponse(_sse_stream(request.question, namespace=request.namespace), media_type="text/event-stream")
