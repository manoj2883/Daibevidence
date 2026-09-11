"""
Grounded, cited, population-aware generation over retrieved PubMed chunks.
Direct Anthropic SDK only — no LangChain. Streams the answer token by token
so the frontend can render it word by word, per spec.
"""
import os
from typing import Any, Dict, Generator, List, Set

import anthropic
from dotenv import load_dotenv

from src.ingest.config import RETRIEVAL_TOP_K
from src.ingest.population import classify_population
from src.rag.cost import assert_chunk_cap, print_prompt_estimate
from src.rag.query_log import log_query_event
from src.rag.retriever import retrieve
from src.rag.types import RetrievedChunk

# Load environment variables
load_dotenv()

REFUSAL_TEXT = "The retrieved literature does not contain sufficient evidence to answer this question."

DISCLAIMER = (
    "This is a summary of published research literature, not medical advice. "
    "It is not a substitute for professional diagnosis, treatment, or medication guidance — "
    "consult a qualified healthcare provider before making any health decisions."
)

SYSTEM_PROMPT = """You are a clinical evidence assistant covering all forms of diabetes \
(type 1, type 2, gestational, and prediabetes) research. You answer strictly from the source \
excerpts below, drawn from PubMed reviews, systematic reviews, meta-analyses, and clinical trials. \
You are not a doctor and nothing you say is medical advice.

Rules, in order of priority:
1. Use ONLY information stated in the source excerpts. Never use outside knowledge, training data, \
or assumptions, even if you believe you know the answer.
2. Cite every factual claim inline by PubMed ID, in the form [PMID: 12345678]. If a claim draws on \
multiple excerpts, cite each one, e.g. [PMID: 11111111][PMID: 22222222].
3. State which diabetes population (type 1, type 2, gestational, prediabetes, or a mix) the evidence \
you cite applies to, as part of the answer itself — not only in a source list.
4. The user's question was automatically classified as being about the "{requested_population}" \
population (this may be "mixed" if the question didn't specify one, or mentioned more than one). \
The retrieved excerpts cover population(s): {retrieved_populations}. If "{requested_population}" is \
not "mixed" and it differs from the retrieved population(s), you MUST explicitly flag this mismatch \
before answering, and make clear the evidence may not generalize to the population actually asked \
about. If "{requested_population}" is "mixed", just state plainly which population(s) the evidence \
covers, with no mismatch to flag.
5. If the excerpts do not contain enough information to answer the question, reply with exactly this \
sentence and nothing else: "{refusal}"
6. Never fabricate a PMID, a statistic, a study finding, or a population.
7. If the question or the evidence concerns diabetes medications, discuss them only in general, \
informational, mechanistic terms (e.g., how a drug class interacts with diet or nutrition). NEVER give \
dosing instructions, prescribing guidance, or advice to start, stop, or adjust a medication.
8. Be precise: prefer specific numbers, effect sizes, and study populations stated in the excerpts \
over vague language.

Source excerpts:
{context}"""


def get_client() -> anthropic.Anthropic:
    """
    Initialize the Anthropic client used for grounded answer generation.
    Fails fast (before any Pinecone call) if the key isn't configured.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY must be set in environment variables.")
    return anthropic.Anthropic(api_key=api_key.strip())


def get_model() -> str:
    return os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")


def format_context(chunks: List[RetrievedChunk]) -> str:
    """
    Render retrieved chunks as a labeled context block for the prompt.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata or {}
        pmid = meta.get("pmid", "unknown")
        title = meta.get("title", "")
        journal = meta.get("journal", "")
        year = meta.get("year", "")
        pub_type = meta.get("publication_type", "")
        population = meta.get("population", "")
        header = (
            f"[Excerpt {i}] PMID {pmid} — {title} ({journal}, {year}) "
            f"| Publication type: {pub_type} | Population: {population}"
        )
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def retrieved_population_set(chunks: List[RetrievedChunk]) -> Set[str]:
    populations = {(chunk.metadata or {}).get("population", "") for chunk in chunks}
    populations.discard("")
    return populations


def retrieved_populations_summary(chunks: List[RetrievedChunk]) -> str:
    populations = retrieved_population_set(chunks)
    return ", ".join(sorted(populations)) if populations else "unknown"


def is_population_mismatch(requested_population: str, retrieved: Set[str]) -> bool:
    """
    Deterministic mismatch flag (independent of what the model writes in
    prose) so the frontend can render the warning banner reliably.
    """
    if requested_population == "mixed" or not retrieved:
        return False
    return requested_population not in retrieved


def _source_payload(chunk: RetrievedChunk) -> Dict[str, Any]:
    meta = chunk.metadata or {}
    return {
        "pmid": meta.get("pmid", ""),
        "title": meta.get("title", ""),
        "journal": meta.get("journal", ""),
        "year": meta.get("year", ""),
        "publication_type": meta.get("publication_type", ""),
        "population": meta.get("population", ""),
        "excerpt": chunk.text,
        "score": chunk.score,
    }


def stream_answer(question: str) -> Generator[Dict[str, Any], None, None]:
    """
    Run the grounded RAG pipeline for one question, yielding a sequence of
    event dicts: {"event": ..., "data": ...}. Possible events:

    - "sources": retrieved chunks + population info, sent once, before generation
    - "token": one text delta of the streaming answer
    - "done": generation finished, carries the disclaimer
    - "refusal": no chunks retrieved — refusal message, no Claude call made
    - "error": something is misconfigured (e.g. missing API key)

    Never sends more than RETRIEVAL_TOP_K chunks to Claude — enforced by a
    hard assertion, not just a printed warning.
    """
    try:
        client = get_client()
    except ValueError as e:
        yield {"event": "error", "data": {"message": str(e)}}
        return

    requested_population = classify_population(question)

    try:
        chunks = retrieve(question)
    except ValueError as e:
        yield {"event": "error", "data": {"message": str(e)}}
        return

    assert_chunk_cap(chunks, RETRIEVAL_TOP_K)

    if not chunks:
        log_query_event(question, requested_population, [], REFUSAL_TEXT)
        yield {
            "event": "refusal",
            "data": {"message": REFUSAL_TEXT, "disclaimer": DISCLAIMER},
        }
        return

    retrieved = retrieved_population_set(chunks)
    mismatch = is_population_mismatch(requested_population, retrieved)

    yield {
        "event": "sources",
        "data": {
            "requested_population": requested_population,
            "retrieved_populations": sorted(retrieved),
            "mismatch": mismatch,
            "sources": [_source_payload(c) for c in chunks],
        },
    }

    context_str = format_context(chunks)
    retrieved_populations_str = retrieved_populations_summary(chunks)
    system_prompt = SYSTEM_PROMPT.format(
        requested_population=requested_population,
        retrieved_populations=retrieved_populations_str,
        refusal=REFUSAL_TEXT,
        context=context_str,
    )

    print_prompt_estimate(f"query: {question[:60]!r}", system_prompt, question)

    full_answer = ""
    try:
        with client.messages.stream(
            model=get_model(),
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
        ) as stream:
            for text in stream.text_stream:
                full_answer += text
                yield {"event": "token", "data": {"text": text}}
    except anthropic.APIError as e:
        # A partial answer may already have streamed — surface the failure
        # explicitly rather than letting the connection die silently.
        yield {"event": "error", "data": {"message": f"Generation failed: {e}"}}
        return

    log_query_event(question, requested_population, chunks, full_answer)

    yield {"event": "done", "data": {"disclaimer": DISCLAIMER}}
