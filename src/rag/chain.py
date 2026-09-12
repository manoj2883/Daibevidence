"""
Grounded, cited, population-aware generation over retrieved PubMed chunks.
Direct Anthropic SDK only — no LangChain. The answer is a JSON array of
per-sentence {sentence, chunk_ids, supported} objects, streamed as one
"sentence" event per completed array element as soon as it parses off the
stream — sentence-by-sentence rather than word-by-word, so citation
attribution is structural (chunk_ids) rather than text markup.
"""
import json
import os
import time
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
# before its sentence array. Kept out of user-visible text — parsed off the
# stream server-side before any sentence content is forwarded.
CONTRA_START = "<<<CONTRADICTIONS>>>"
CONTRA_END = "<<<END_CONTRADICTIONS>>>"

# If the model never emits a complete contradiction block within this many
# buffered characters, give up waiting and treat everything buffered so far
# as the start of its answer content (handles a bare refusal-sentence
# response, or plain non-compliance).
MAX_PREAMBLE_BUFFER = 6000

SYSTEM_PROMPT = """You are a clinical evidence assistant covering all forms of diabetes \
(type 1, type 2, gestational, and prediabetes) research. You answer strictly from the source \
excerpts below. Excerpts come from two tiers, each labeled SOURCE TYPE in its header: "evidence" \
(PubMed reviews, systematic reviews, meta-analyses, and clinical trials) and "background" \
(patient-education pages from ADA, NIDDK, or CDC, covering definitions, diagnostic criteria, and \
general disease mechanism — not research findings). You are not a doctor and nothing you say is \
medical advice.

Rules, in order of priority:
1. Use ONLY information stated in the source excerpts. Never use outside knowledge, training data, \
or assumptions, even if you believe you know the answer.
2. A "background" excerpt may ONLY support a sentence that defines or explains something — what a \
condition is, how it's diagnosed, or how it generally works. A "background" excerpt must NEVER be the \
sole or partial support for a sentence that claims an intervention works, states or implies an effect \
size, makes a comparison, or reports any research finding — that requires an "evidence" excerpt. If a \
sentence mixes a definitional point with a research claim, split it into separate sentences so each \
one's chunk_ids correctly reflect what actually supports it.
3. Output your answer as a single JSON array, one object per sentence, in order. Each object has \
exactly these keys: "sentence" (one complete sentence of your answer, plain text — do NOT include \
bracketed PubMed citations like [PMID: 123] in this text; attribution is carried entirely by \
chunk_ids, not by markup in the sentence), "chunk_ids" (array of the [Excerpt N] numbers from below \
that support this specific sentence's claim, e.g. [1, 3], or [] if the sentence is pure transition or \
framing with no factual claim), and "supported" (true only if chunk_ids is non-empty and the sentence \
genuinely follows from those excerpts, respecting rule 2's tier restriction). If a specific clinical \
claim cannot be attributed to any excerpt, do not include it as a sentence at all — never present an \
unsupported clinical claim, whether marked or not.
4. Before the JSON array, compare the "evidence"-tier excerpts for genuine contradictions — cases \
where two excerpts report conflicting findings on the same specific claim (not just different topics, \
and not a population difference — that is handled separately). Background excerpts don't report \
research findings, so they cannot contradict anything under this rule. Report this as a JSON preamble, \
exactly as follows, before the sentence array: a line containing exactly "{contra_start}", then a JSON \
array (or "[]" if none), then a line containing exactly "{contra_end}". Each array entry must have \
exactly these keys: "excerpt_indices" (array of the [Excerpt N] numbers involved, e.g. [1, 3]), "claim" \
(the specific point they disagree on), "position_a" (what the first excerpt found), "position_b" (what \
the other found), and "differs_by" (what differs between the studies that could explain the \
disagreement — e.g. population, study design, duration, sample size). If you report a contradiction \
here, your sentence array below MUST present both positions rather than silently picking one. \
Exception: if you are refusing per rule 7 below, skip this preamble and the JSON array entirely and \
output only the refusal sentence as plain text.
5. State which diabetes population (type 1, type 2, gestational, prediabetes, or a mix) the evidence \
you cite applies to, as one of the sentences in your answer — not only in a source list.
6. The user's question was automatically classified as being about the "{requested_population}" \
population (this may be "mixed" if the question didn't specify one, or mentioned more than one). \
The retrieved excerpts cover population(s): {retrieved_populations}. If "{requested_population}" is \
not "mixed" and it differs from the retrieved population(s), you MUST explicitly flag this mismatch \
as one of your sentences before answering, and make clear the evidence may not generalize to the \
population actually asked about. If "{requested_population}" is "mixed", just state plainly which \
population(s) the evidence covers, with no mismatch to flag. Do not report this as a contradiction \
under rule 4 — population mismatch is a distinct thing from two studies disagreeing.
7. If the excerpts do not contain enough information to answer the question, reply with exactly this \
sentence and nothing else (no contradiction preamble, no JSON array): "{refusal}"
8. Never fabricate a PMID, a statistic, a study finding, or a population.
9. If the question or the evidence concerns diabetes medications, discuss them only in general, \
informational, mechanistic terms (e.g., how a drug class interacts with diet or nutrition). NEVER give \
dosing instructions, prescribing guidance, or advice to start, stop, or adjust a medication.
10. Be precise: prefer specific numbers, effect sizes, and study populations stated in the excerpts \
over vague language — from "evidence" excerpts only, per rule 2.

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
    Render retrieved chunks as a labeled context block for the prompt. Every
    header leads with SOURCE TYPE so the model can never miss which tier an
    excerpt comes from — evidence (PubMed) vs. background (ADA/NIDDK/CDC
    patient education), which the SYSTEM_PROMPT restricts to definitional
    use only.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata or {}
        population = meta.get("population", "")
        source_type = meta.get("source_type", "evidence")

        if source_type == "background":
            publisher = meta.get("publisher", "")
            title = meta.get("title", "")
            header = (
                f"[Excerpt {i}] SOURCE TYPE: background | Publisher: {publisher} — {title} "
                f"| Population: {population}"
            )
        else:
            pmid = meta.get("pmid", "unknown")
            title = meta.get("title", "")
            journal = meta.get("journal", "")
            year = meta.get("year", "")
            pub_type = meta.get("publication_type", "")
            header = (
                f"[Excerpt {i}] SOURCE TYPE: evidence | PMID {pmid} — {title} ({journal}, {year}) "
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
        "source_type": meta.get("source_type", "evidence"),
        "pmid": meta.get("pmid", ""),
        "title": meta.get("title", ""),
        "journal": meta.get("journal", ""),
        "year": meta.get("year", ""),
        "publication_type": meta.get("publication_type", ""),
        "publisher": meta.get("publisher", ""),
        "source_url": meta.get("source_url", ""),
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


def _pmids_for_indices(indices: List[Any], chunks: List[RetrievedChunk]) -> List[str]:
    """
    Resolve [Excerpt N] indices to PMIDs, silently dropping any index that
    isn't a valid in-range int rather than failing the whole lookup.
    """
    pmids = []
    for idx in indices:
        if isinstance(idx, int) and 1 <= idx <= len(chunks):
            pmids.append((chunks[idx - 1].metadata or {}).get("pmid", ""))
    return pmids


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
    where `rest` is everything after the end marker — never crashes on
    malformed JSON, just returns [] for it.
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


def find_complete_json_objects(buffer: str, start: int = 0) -> Tuple[List[str], int]:
    """
    Pure incremental scanner over a streaming text buffer that's expected
    to hold a JSON array of objects. Scans from `start`, skipping
    whitespace, commas, and the array brackets, and returns the raw text of
    every *complete* top-level {...} object found — respecting string
    literals and escapes, so a brace inside a sentence's text never
    confuses the scan — plus the position to resume scanning from next
    time (so re-scanning never repeats work as more text streams in).

    An incomplete trailing object (still streaming in) is left for the
    next call; a genuinely non-JSON buffer (e.g. plain refusal text)
    simply yields no objects, forever, without raising.
    """
    i, n = start, len(buffer)
    found = []
    while i < n:
        c = buffer[i]
        if c in " \t\r\n,[]":
            i += 1
            continue
        if c != "{":
            break  # not JSON-array-shaped content — stop scanning here

        depth, in_string, escape = 0, False, False
        j = i
        while j < n:
            cj = buffer[j]
            if in_string:
                if escape:
                    escape = False
                elif cj == "\\":
                    escape = True
                elif cj == '"':
                    in_string = False
            else:
                if cj == '"':
                    in_string = True
                elif cj == "{":
                    depth += 1
                elif cj == "}":
                    depth -= 1
                    if depth == 0:
                        break
            j += 1

        if depth != 0:
            break  # object not complete yet — wait for more streamed text

        found.append(buffer[i:j + 1])
        i = j + 1

    return found, i


def _validate_sentence(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Coerce a parsed JSON object into the {sentence, chunk_ids, supported}
    shape, or return None if it's missing the one thing that actually
    matters (sentence text) — never propagate a malformed entry.
    """
    if not isinstance(obj, dict):
        return None
    sentence = obj.get("sentence")
    if not isinstance(sentence, str) or not sentence.strip():
        return None
    chunk_ids = obj.get("chunk_ids")
    chunk_ids = [c for c in chunk_ids if isinstance(c, int)] if isinstance(chunk_ids, list) else []
    supported = obj.get("supported")
    if not isinstance(supported, bool):
        supported = bool(chunk_ids)
    return {"sentence": sentence.strip(), "chunk_ids": chunk_ids, "supported": supported}


def _sentence_payload(sentence: Dict[str, Any], chunks: List[RetrievedChunk]) -> Dict[str, Any]:
    return {
        "sentence": sentence["sentence"],
        "chunk_ids": sentence["chunk_ids"],
        "pmids": _pmids_for_indices(sentence["chunk_ids"], chunks),
        "supported": sentence["supported"],
    }


def compute_groundedness(sentences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    The headline metric: share of sentences with at least one supporting
    chunk. Uses chunk_ids presence (objective, checkable) rather than the
    model's self-reported `supported` flag as the authoritative signal.
    """
    total = len(sentences)
    supported = sum(1 for s in sentences if s.get("chunk_ids"))
    return {
        "groundedness": round(supported / total, 4) if total else None,
        "supported_sentences": supported,
        "total_sentences": total,
    }


def stream_answer(question: str) -> Generator[Dict[str, Any], None, None]:
    """
    Run the grounded RAG pipeline for one question, yielding a sequence of
    event dicts: {"event": ..., "data": ...}. Possible events:

    - "sources": retrieved chunks + population info + retrieval state
      ("answered" or "answered_low_confidence") + score distribution, sent
      once, before generation
    - "contradictions": conflicting findings detected across excerpts (an
      empty list if none), sent once, before any "sentence" events
    - "sentence": one {sentence, chunk_ids, pmids, supported} object, sent
      as soon as it completes in the stream
    - "done": generation finished, carries the disclaimer and the
      groundedness summary (share of sentences with a supporting chunk)
    - "refusal": nothing cleared the similarity floor — refusal message,
      closest scores found, what the system covers, and the score
      distribution. No Claude call made.
    - "error": something is misconfigured (e.g. missing API key) or
      generation itself failed

    Retrieves a wide candidate pool (RETRIEVAL_CANDIDATE_K) and lets the
    similarity floor decide how many survive, rather than always sending a
    fixed top-k. Never sends more than RETRIEVAL_CANDIDATE_K chunks to
    Claude — enforced by a hard assertion, not just a printed warning.
    """
    t_start = time.monotonic()

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

    t_after_retrieval = time.monotonic()
    retrieval_ms = round((t_after_retrieval - t_start) * 1000, 1)

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
                "timing_ms": {"retrieval": retrieval_ms},
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
            "timing_ms": {"retrieval": retrieval_ms},
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

    preamble_buffer = ""
    contradictions_resolved = False
    sentence_buffer = ""
    sentence_scan_pos = 0
    full_sentences: List[Dict[str, Any]] = []

    def _emit_new_sentences():
        nonlocal sentence_scan_pos
        objs, sentence_scan_pos = find_complete_json_objects(sentence_buffer, sentence_scan_pos)
        events = []
        for obj_text in objs:
            try:
                parsed = json.loads(obj_text)
            except json.JSONDecodeError:
                continue
            validated = _validate_sentence(parsed)
            if validated is None:
                continue
            full_sentences.append(validated)
            events.append({"event": "sentence", "data": _sentence_payload(validated, chunks)})
        return events

    try:
        with client.messages.stream(
            model=get_model(),
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
        ) as stream:
            for text in stream.text_stream:
                if contradictions_resolved:
                    sentence_buffer += text
                    for ev in _emit_new_sentences():
                        yield ev
                    continue

                preamble_buffer += text
                result = extract_contradiction_block(preamble_buffer)
                if result is not None:
                    contradictions, remainder = result
                    yield {
                        "event": "contradictions",
                        "data": {
                            "contradictions": _resolve_contradiction_pmids(contradictions, chunks),
                            "timing_ms": {"contradiction_check": round((time.monotonic() - t_after_retrieval) * 1000, 1)},
                        },
                    }
                    contradictions_resolved = True
                    preamble_buffer = ""
                    if remainder:
                        sentence_buffer += remainder
                        for ev in _emit_new_sentences():
                            yield ev
                elif len(preamble_buffer) > MAX_PREAMBLE_BUFFER:
                    # Model didn't emit the expected format — stop waiting,
                    # treat everything buffered so far as the start of its
                    # answer content (e.g. a bare refusal-sentence response).
                    yield {
                        "event": "contradictions",
                        "data": {
                            "contradictions": [],
                            "timing_ms": {"contradiction_check": round((time.monotonic() - t_after_retrieval) * 1000, 1)},
                        },
                    }
                    contradictions_resolved = True
                    sentence_buffer += preamble_buffer
                    preamble_buffer = ""
                    for ev in _emit_new_sentences():
                        yield ev

        if not contradictions_resolved:
            # Stream ended entirely within the buffering window (very short
            # response, e.g. exactly the refusal sentence).
            yield {
                "event": "contradictions",
                "data": {
                    "contradictions": [],
                    "timing_ms": {"contradiction_check": round((time.monotonic() - t_after_retrieval) * 1000, 1)},
                },
            }
            sentence_buffer += preamble_buffer
            for ev in _emit_new_sentences():
                yield ev

        if not full_sentences:
            # Nothing parsed as a valid sentence object at all — either the
            # model didn't comply with the JSON format, or (normally) this
            # is a bare refusal-sentence response. Either way, the API
            # contract always returns at least one sentence object.
            leftover = sentence_buffer.strip()
            if leftover:
                synthetic = {"sentence": leftover, "chunk_ids": [], "supported": False}
                full_sentences.append(synthetic)
                yield {"event": "sentence", "data": _sentence_payload(synthetic, chunks)}
    except anthropic.APIError as e:
        # A partial answer may already have streamed — surface the failure
        # explicitly rather than letting the connection die silently.
        yield {"event": "error", "data": {"message": f"Generation failed: {e}"}}
        return

    full_answer = " ".join(s["sentence"] for s in full_sentences)
    groundedness = compute_groundedness(full_sentences)
    t_end = time.monotonic()
    timing_ms = {
        "retrieval": retrieval_ms,
        "generation": round((t_end - t_after_retrieval) * 1000, 1),
        "total": round((t_end - t_start) * 1000, 1),
    }

    log_query_event(
        question, requested_population, chunks, full_answer,
        state=decision.state, sentences=full_sentences, groundedness=groundedness["groundedness"],
    )

    yield {"event": "done", "data": {"disclaimer": DISCLAIMER, "groundedness": groundedness, "timing_ms": timing_ms}}
