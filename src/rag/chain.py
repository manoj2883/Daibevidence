"""
Grounded, cited, population-aware generation over retrieved PubMed chunks.
Direct Anthropic SDK only — no LangChain. Streams the answer token by token
so the frontend can render it word by word, per spec.
"""
import json
import os
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import anthropic
from dotenv import load_dotenv

from src.ingest.config import RETRIEVAL_CANDIDATE_K
from src.ingest.population import classify_population
from src.rag.cost import assert_chunk_cap, print_prompt_estimate
from src.rag.query_log import log_query_event
from src.rag.retriever import retrieve_with_floor
from src.rag.types import RetrievalDecision, RetrievedChunk

# Load environment variables
load_dotenv()

REFUSAL_TEXT = "The retrieved literature does not contain sufficient evidence to answer this question."

SCOPE_DESCRIPTION = (
    "This system answers questions on four topics, across all diabetes types (type 1, "
    "type 2, gestational, and prediabetes): diet and nutrition, glycemic control, body "
    "composition and weight, and how diet interacts with diabetes medication."
)

DISCLAIMER = (
    "This is a summary of published research literature, not medical advice. "
    "It is not a substitute for professional diagnosis, treatment, or medication guidance — "
    "consult a qualified healthcare provider before making any health decisions."
)

# Markers delimiting the contradiction-check JSON block the model must emit
# before its prose answer. Kept out of user-visible text — parsed off the
# stream server-side before any prose is forwarded as "token" events.
CONTRA_START = "<<<CONTRADICTIONS>>>"
CONTRA_END = "<<<END_CONTRADICTIONS>>>"

# If the model never emits a complete contradiction block within this many
# buffered characters, give up waiting and treat everything buffered so far
# as prose (handles a refusal-sentence response, or plain non-compliance).
MAX_PREAMBLE_BUFFER = 6000

SYSTEM_PROMPT = """You are a clinical evidence assistant covering all forms of diabetes \
(type 1, type 2, gestational, and prediabetes) research. You answer strictly from the source \
excerpts below, drawn from PubMed reviews, systematic reviews, meta-analyses, and clinical trials. \
You are not a doctor and nothing you say is medical advice.

Rules, in order of priority:
1. Use ONLY information stated in the source excerpts. Never use outside knowledge, training data, \
or assumptions, even if you believe you know the answer.
2. Cite every factual claim inline by PubMed ID, in the form [PMID: 12345678]. If a claim draws on \
multiple excerpts, cite each one, e.g. [PMID: 11111111][PMID: 22222222].
3. Before writing your answer, compare the source excerpts for genuine contradictions — cases where \
two excerpts report conflicting findings on the same specific claim (not just different topics, and \
not a population difference — that is handled separately). Report this as a JSON preamble, exactly as \
follows, before any other text: a line containing exactly "{contra_start}", then a JSON array (or "[]" \
if none), then a line containing exactly "{contra_end}". Each array entry must have exactly these keys: \
"excerpt_indices" (array of the [Excerpt N] numbers involved, e.g. [1, 3]), "claim" (the specific point \
they disagree on), "position_a" (what the first excerpt found), "position_b" (what the other found), and \
"differs_by" (what differs between the studies that could explain the disagreement — e.g. population, \
study design, duration, sample size). If you report a contradiction here, your prose answer below MUST \
present both positions rather than silently picking one. \
Exception: if you are refusing per rule 6 below, skip this preamble entirely and output only the refusal \
sentence.
4. State which diabetes population (type 1, type 2, gestational, prediabetes, or a mix) the evidence \
you cite applies to, as part of the answer itself — not only in a source list.
5. The user's question was automatically classified as being about the "{requested_population}" \
population (this may be "mixed" if the question didn't specify one, or mentioned more than one). \
The retrieved excerpts cover population(s): {retrieved_populations}. If "{requested_population}" is \
not "mixed" and it differs from the retrieved population(s), you MUST explicitly flag this mismatch \
before answering, and make clear the evidence may not generalize to the population actually asked \
about. If "{requested_population}" is "mixed", just state plainly which population(s) the evidence \
covers, with no mismatch to flag. Do not report this as a contradiction under rule 3 — population \
mismatch is a distinct thing from two studies disagreeing.
6. If the excerpts do not contain enough information to answer the question, reply with exactly this \
sentence and nothing else (no contradiction preamble): "{refusal}"
7. Never fabricate a PMID, a statistic, a study finding, or a population.
8. If the question or the evidence concerns diabetes medications, discuss them only in general, \
informational, mechanistic terms (e.g., how a drug class interacts with diet or nutrition). NEVER give \
dosing instructions, prescribing guidance, or advice to start, stop, or adjust a medication.
9. Be precise: prefer specific numbers, effect sizes, and study populations stated in the excerpts \
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


def _score_distribution_payload(decision: RetrievalDecision) -> Dict[str, Any]:
    return {
        "floor": decision.floor,
        "low_confidence_margin": decision.low_confidence_margin,
        "candidate_scores": decision.candidate_scores,
        "surviving_count": len(decision.surviving_chunks),
        "top_score": decision.top_score,
    }


def _resolve_contradiction_pmids(contradictions: List[Dict[str, Any]], chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
    """
    Attach the actual PMIDs behind each [Excerpt N] reference, and drop any
    entry with malformed or out-of-range indices rather than letting a
    non-compliant model response propagate garbage to the frontend.
    """
    resolved = []
    for entry in contradictions:
        if not isinstance(entry, dict):
            continue
        indices = entry.get("excerpt_indices")
        if not isinstance(indices, list) or not indices:
            continue
        pmids = []
        valid = True
        for idx in indices:
            if not isinstance(idx, int) or not (1 <= idx <= len(chunks)):
                valid = False
                break
            pmids.append((chunks[idx - 1].metadata or {}).get("pmid", ""))
        if not valid:
            continue
        resolved.append({
            "excerpt_indices": indices,
            "pmids": pmids,
            "claim": entry.get("claim", ""),
            "position_a": entry.get("position_a", ""),
            "position_b": entry.get("position_b", ""),
            "differs_by": entry.get("differs_by", ""),
        })
    return resolved


def extract_contradiction_block(buffer: str) -> Optional[Tuple[List[Dict[str, Any]], str]]:
    """
    Pure parser over a streaming text buffer. Returns None if the buffer
    doesn't yet contain a complete CONTRA_START...CONTRA_END block (caller
    should keep buffering). Once complete, returns (contradictions, rest)
    where `rest` is everything after the end marker (the start of the prose
    answer) — never crashes on malformed JSON, just returns [] for it.
    """
    end_idx = buffer.find(CONTRA_END)
    if end_idx == -1:
        return None

    start_idx = buffer.find(CONTRA_START)
    rest = buffer[end_idx + len(CONTRA_END):]

    if start_idx == -1:
        return [], rest

    json_str = buffer[start_idx + len(CONTRA_START):end_idx].strip()
    if not json_str:
        return [], rest

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        return [], rest

    return (parsed if isinstance(parsed, list) else []), rest


def stream_answer(question: str) -> Generator[Dict[str, Any], None, None]:
    """
    Run the grounded RAG pipeline for one question, yielding a sequence of
    event dicts: {"event": ..., "data": ...}. Possible events:

    - "sources": retrieved chunks + population info + retrieval state
      ("answered" or "answered_low_confidence") + score distribution, sent
      once, before generation
    - "contradictions": conflicting findings detected across excerpts (an
      empty list if none), sent once, before any "token" events
    - "token": one text delta of the streaming prose answer
    - "done": generation finished, carries the disclaimer
    - "refusal": nothing cleared the similarity floor — refusal message,
      closest scores found, what the system covers, and the score
      distribution. No Claude call made.
    - "error": something is misconfigured (e.g. missing API key)

    Retrieves a wide candidate pool (RETRIEVAL_CANDIDATE_K) and lets the
    similarity floor decide how many survive, rather than always sending a
    fixed top-k. Never sends more than RETRIEVAL_CANDIDATE_K chunks to
    Claude — enforced by a hard assertion, not just a printed warning.
    """
    try:
        client = get_client()
    except ValueError as e:
        yield {"event": "error", "data": {"message": str(e)}}
        return

    requested_population = classify_population(question)

    try:
        decision = retrieve_with_floor(question)
    except ValueError as e:
        yield {"event": "error", "data": {"message": str(e)}}
        return

    assert_chunk_cap(decision.surviving_chunks, RETRIEVAL_CANDIDATE_K)

    if decision.state == "refused":
        closest_scores = decision.candidate_scores[:3]
        log_query_event(question, requested_population, [], REFUSAL_TEXT, state="refused")
        yield {
            "event": "refusal",
            "data": {
                "message": REFUSAL_TEXT,
                "scope_description": SCOPE_DESCRIPTION,
                "closest_scores": closest_scores,
                "score_distribution": _score_distribution_payload(decision),
                "disclaimer": DISCLAIMER,
            },
        }
        return

    chunks = decision.surviving_chunks
    retrieved = retrieved_population_set(chunks)
    mismatch = is_population_mismatch(requested_population, retrieved)

    yield {
        "event": "sources",
        "data": {
            "state": decision.state,
            "requested_population": requested_population,
            "retrieved_populations": sorted(retrieved),
            "mismatch": mismatch,
            "score_distribution": _score_distribution_payload(decision),
            "sources": [_source_payload(c) for c in chunks],
        },
    }

    context_str = format_context(chunks)
    retrieved_populations_str = retrieved_populations_summary(chunks)
    system_prompt = SYSTEM_PROMPT.format(
        requested_population=requested_population,
        retrieved_populations=retrieved_populations_str,
        refusal=REFUSAL_TEXT,
        contra_start=CONTRA_START,
        contra_end=CONTRA_END,
        context=context_str,
    )

    print_prompt_estimate(f"query: {question[:60]!r}", system_prompt, question)

    full_answer = ""
    preamble_buffer = ""
    contradictions_resolved = False

    def _flush_preamble_as_prose():
        nonlocal full_answer
        if preamble_buffer:
            full_answer += preamble_buffer
            return {"event": "token", "data": {"text": preamble_buffer}}
        return None

    try:
        with client.messages.stream(
            model=get_model(),
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
        ) as stream:
            for text in stream.text_stream:
                if contradictions_resolved:
                    full_answer += text
                    yield {"event": "token", "data": {"text": text}}
                    continue

                preamble_buffer += text
                result = extract_contradiction_block(preamble_buffer)
                if result is not None:
                    contradictions, remainder = result
                    yield {
                        "event": "contradictions",
                        "data": {"contradictions": _resolve_contradiction_pmids(contradictions, chunks)},
                    }
                    contradictions_resolved = True
                    preamble_buffer = ""
                    if remainder:
                        full_answer += remainder
                        yield {"event": "token", "data": {"text": remainder}}
                elif len(preamble_buffer) > MAX_PREAMBLE_BUFFER:
                    # Model didn't emit the expected format — stop waiting,
                    # treat everything buffered so far as prose (e.g. a
                    # refusal-sentence response), no contradictions.
                    yield {"event": "contradictions", "data": {"contradictions": []}}
                    contradictions_resolved = True
                    flushed = _flush_preamble_as_prose()
                    preamble_buffer = ""
                    if flushed:
                        yield flushed

        if not contradictions_resolved:
            # Stream ended entirely within the buffering window (very short
            # response, e.g. exactly the refusal sentence).
            yield {"event": "contradictions", "data": {"contradictions": []}}
            flushed = _flush_preamble_as_prose()
            if flushed:
                yield flushed
    except anthropic.APIError as e:
        # A partial answer may already have streamed — surface the failure
        # explicitly rather than letting the connection die silently.
        yield {"event": "error", "data": {"message": f"Generation failed: {e}"}}
        return

    log_query_event(question, requested_population, chunks, full_answer, state=decision.state)

    yield {"event": "done", "data": {"disclaimer": DISCLAIMER}}
