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
from src.rag.judge import judge_answerability
from src.rag.query_classifier import classify_and_decompose, overall_question_type
from src.rag.query_log import log_query_event
from src.rag.retriever import (
    decide_retrieval_state,
    get_candidate_k,
    get_retriever_k,
    get_similarity_floor,
    get_similarity_floor_background,
    resolve_namespace,
    retrieve,
)
from src.rag.types import RetrievalDecision, RetrievedChunk

# Load environment variables
load_dotenv()

OUT_OF_SCOPE_TEXT = "The retrieved literature does not contain sufficient evidence to answer this question."

# The answerability judge's refusal message — distinct from OUT_OF_SCOPE_TEXT
# above, which now only fires in the defensive case of zero retrieved
# candidates (see _retrieve_for_question / stream_answer).
JUDGE_REFUSAL_TEXT = "I couldn't find evidence in the literature I have that answers this question."

# How many top-scoring chunks the answerability judge sees. Deliberately
# small: the judge's job is a quick answerability check, not a full review
# of every candidate that cleared retrieval.
JUDGE_CHUNK_LIMIT = 5

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
# as the start of its answer content — a defensive fallback for format
# non-compliance, not a designed path (the prompt always expects JSON now).
MAX_PREAMBLE_BUFFER = 6000

SYSTEM_PROMPT = """You are a clinical evidence assistant covering all forms of diabetes \
(type 1, type 2, gestational, and prediabetes) research. You answer strictly from the source \
excerpts below. Excerpts come from two tiers, each labeled SOURCE TYPE in its header: "background" \
(patient-education pages from ADA or NIDDK, covering definitions, diagnostic criteria, and general \
disease mechanism — not research findings) and "study" (PubMed reviews, systematic reviews, \
meta-analyses, and clinical trials). You are not a doctor and nothing you say is medical advice.

Write for a general reader: plain language at roughly an 8th-grade reading level. Use a technical \
term only when a plain word would lose meaning, and keep sentences short.

Rules, in order of priority:
1. Use ONLY information stated in the source excerpts. Never use outside knowledge, training data, \
or assumptions, even if you believe you know the answer.
2. A "background" excerpt may ONLY support a sentence that defines or explains something — what a \
condition is, how it's diagnosed, or how it generally works. A "background" excerpt must NEVER be the \
sole or partial support for a sentence that claims an intervention works, states or implies an effect \
size, makes a comparison, or reports any research finding — that requires a "study" excerpt. If a \
sentence mixes a definitional point with a research claim, split it into separate sentences so each \
one's chunk_ids correctly reflect what actually supports it.
3. Output your answer as a single JSON array, one object per sentence, in order. Each object has \
exactly these keys: "sentence" (one complete sentence of your answer, plain text — do NOT include \
bracketed PubMed citations like [PMID: 123] in this text; attribution is carried entirely by \
chunk_ids, not by markup in the sentence), "chunk_ids" (array of the [Excerpt N] numbers from below \
that support this specific sentence's claim, e.g. [1, 3], or [] if the sentence is pure transition or \
framing with no factual claim), "supported" (true only if chunk_ids is non-empty and the sentence \
genuinely follows from those excerpts, respecting rule 2's tier restriction), and "new_paragraph" \
(boolean — true if this sentence starts a new paragraph). Keep paragraphs short and focused on one \
idea; start a new paragraph at each real shift in idea, and always at the shift from \
background/definitional content to study/evidence content (or back). The very first sentence is its \
own paragraph regardless of this flag's value. If a specific clinical claim cannot be attributed to \
any excerpt, do not include it as a sentence at all — never present an unsupported clinical claim, \
whether marked or not.
4. Before the JSON array, compare the "study"-tier excerpts for genuine contradictions — cases where \
two excerpts report conflicting findings on the same specific claim (not just different topics, and \
not a population difference — that is handled separately). Background excerpts don't report research \
findings, so they cannot contradict anything under this rule. Report this as a JSON preamble, exactly \
as follows, before the sentence array: a line containing exactly "{contra_start}", then a JSON array \
(or "[]" if none), then a line containing exactly "{contra_end}". Each array entry must have exactly \
these keys: "excerpt_indices" (array of the [Excerpt N] numbers involved, e.g. [1, 3]), "claim" (the \
specific point they disagree on), "position_a" (what the first excerpt found), "position_b" (what the \
other found), and "differs_by" (what differs between the studies that could explain the disagreement \
— e.g. population, study design, duration, sample size). If you report a contradiction here, your \
sentence array below MUST present both positions rather than silently picking one.
5. Lead with the direct answer — the first sentence answers the question, never a caveat. When the \
question has both a definitional part and a research part (see "Parts to answer, in order" below if \
the question was split), answer the definitional part first from background excerpts, then what the \
studies show from study excerpts, in that order. State which diabetes population (type 1, type 2, \
gestational, prediabetes, or a mix) the evidence you cite applies to, as one of your sentences — not \
only in a source list. Every caveat — a population mismatch (rule 6), a part of the question that \
can't be answered (rule 7), reduced confidence, anything hedging the answer — comes AFTER the direct \
answer. Never open with a caveat.
6. The user's question was automatically classified as being about the following population(s): \
"{requested_population}" ("mixed" means the question didn't name a specific one). The retrieved \
excerpts cover population(s): {retrieved_populations}. Flag a population mismatch — as a caveat \
sentence placed AFTER your main answer, per rule 5, never as the opening sentence — only when \
"{requested_population}" is not "mixed" AND none of the named population(s) are among the retrieved \
population(s) AND the retrieved population(s) are not "mixed" (general/all-population content covers \
whatever was asked, so that is not a mismatch). Otherwise, just state plainly which population(s) the \
evidence covers, with no mismatch to flag. Do not report a population mismatch as a contradiction \
under rule 4 — they are different things.
7. If no excerpt actually answers a specific claim in the question — even if some excerpts cleared \
retrieval — still output the normal contradiction preamble and JSON array. Include any genuinely \
relevant definitional sentences from background excerpts if available (each correctly citing their \
chunk_ids, per rule 2), then end with exactly one final sentence, in plain language, stating that no \
study in the retrieved literature addresses [restate the specific claim] — give that sentence \
chunk_ids: [] and supported: false. Per rule 5, this is a closing statement, not a preamble: it comes \
last, after everything that IS answered. This rule applies EVEN WHEN the question also has a \
definitional part that IS answered from background excerpts: answering the definitional part does not \
excuse you from this final sentence for the part that has no study behind it. A population statement \
(rule 6) is never a substitute for this — they answer a different question ("who does the evidence \
cover") than this one ("is there a study on this specific claim at all"). Never stretch an excerpt to \
appear to answer something it doesn't just to avoid saying so.
8. Never fabricate a PMID, a statistic, a study finding, or a population.
9. If the question or the evidence concerns diabetes medications, discuss them only in general, \
informational, mechanistic terms (e.g., how a drug class interacts with diet or nutrition). NEVER give \
dosing instructions, prescribing guidance, or advice to start, stop, or adjust a medication.
10. Be precise: prefer specific numbers, effect sizes, and study populations stated in the excerpts \
over vague language — from "study" excerpts only, per rule 2.

{sub_question_note}Source excerpts:
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
    excerpt comes from — "study" (PubMed) vs. "background" (ADA/NIDDK
    patient education), which the SYSTEM_PROMPT restricts to definitional
    use only. The label shown here is "study" (matching SYSTEM_PROMPT's
    vocabulary and the UI's citation-chip label); the underlying metadata
    value stays "evidence" internally (see _is_evidence_chunk) — renaming
    that would mean re-tagging every already-uploaded vector for no
    behavioral benefit, so only the word shown to the model/user changed.
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
                f"[Excerpt {i}] SOURCE TYPE: study | PMID {pmid} — {title} ({journal}, {year}) "
                f"| Publication type: {pub_type} | Population: {population}"
            )
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def _chunk_identity(chunk: RetrievedChunk) -> Tuple[str, Any, Any]:
    """
    A stable de-duplication key for merging chunks retrieved across
    multiple sub-questions (see _retrieve_for_question) — two sub-questions
    of the same overall type could otherwise both surface the same chunk.
    """
    meta = chunk.metadata or {}
    if meta.get("source_type") == "background":
        return ("background", meta.get("publisher"), (meta.get("title"), meta.get("chunk_index")))
    return ("evidence", meta.get("pmid"), meta.get("chunk_index"))


def _tier_for_question_type(question_type: str) -> str:
    return "background" if question_type == "definitional" else "study"


def _retrieve_for_question(
    question: str, namespace: Optional[str], retrieval_mode: str = "topk"
) -> Tuple[RetrievalDecision, List[Dict[str, Any]]]:
    """
    Classifies `question` — possibly decomposing it into sub-questions, see
    src.rag.query_classifier — and retrieves each sub-question against its
    own tier only: a definitional sub-question retrieves from the
    background tier only, an evidence-seeking one from the study tier only
    (rather than letting the two tiers compete on raw similarity score
    within one blended top-k, which is how a purely definitional question
    like "what is X" could end up citing a study excerpt that happens to
    score higher than the actual definition).

    Merges the per-sub-question results into one RetrievalDecision whose
    surviving_chunks are ordered with every definitional sub-question's
    chunks first, then every evidence-seeking sub-question's — since
    classify_and_decompose already returns sub-questions in that order,
    concatenating preserves it. format_context() renders chunks in list
    order, so the excerpt list Claude sees — and therefore the order rule 5
    asks it to follow — is enforced structurally, not just by instruction.

    retrieval_mode selects which chunks become surviving_chunks:
    - "topk" (default, production behavior): the top get_retriever_k()
      candidates by raw score, regardless of whether they cleared the
      similarity floor — docs/findings-retrieval-floor.md found the floor
      can't separate topically-related-but-unanswerable questions from
      genuinely answerable ones, so it no longer gates which chunks reach
      the judge/generation stage (src.rag.judge does that job now).
    - "floor": only candidates that cleared the similarity floor (the
      pre-judge behavior) — used exclusively by diagnostic/eval scripts
      reconstructing the old floor-gated pipeline for comparison (see
      scripts/run_old_pipeline_full.py); never used by the live /query path.
    Either way, candidate_scores/floor/low_confidence_margin on the merged
    RetrievalDecision are always computed over the full candidate pool via
    decide_retrieval_state, driving the retrieval inspector and eval
    logging unchanged regardless of mode.

    Returns (merged_decision, classified_sub_questions) — the caller needs
    the sub-question breakdown to build the prompt's "parts to answer" note.
    """
    if retrieval_mode not in ("topk", "floor"):
        raise ValueError(f"retrieval_mode must be 'topk' or 'floor', got {retrieval_mode!r}")

    classified = classify_and_decompose(question)

    seen = set()
    merged_chunks: List[RetrievedChunk] = []
    all_candidate_scores: List[float] = []
    last_floor: Optional[float] = None
    last_background_floor: Optional[float] = None
    last_margin: Optional[float] = None

    for item in classified:
        tier = _tier_for_question_type(item["type"])
        candidates = retrieve(item["question"], k=get_candidate_k(), namespace=namespace, tier=tier)
        scored = decide_retrieval_state(
            candidates, get_similarity_floor(), background_floor=get_similarity_floor_background()
        )
        last_floor = scored.floor
        last_background_floor = scored.background_floor
        last_margin = scored.low_confidence_margin
        all_candidate_scores.extend(scored.candidate_scores)

        selected = scored.surviving_chunks if retrieval_mode == "floor" else candidates[: get_retriever_k()]
        for chunk in selected:
            key = _chunk_identity(chunk)
            if key in seen:
                continue
            seen.add(key)
            merged_chunks.append(chunk)

    merged = RetrievalDecision(
        state="out_of_scope" if not merged_chunks else "answered",
        surviving_chunks=merged_chunks,
        candidate_scores=all_candidate_scores,
        floor=last_floor,
        background_floor=last_background_floor,
        low_confidence_margin=last_margin,
    )
    return merged, classified


def _sub_question_note(classified: List[Dict[str, Any]]) -> str:
    """
    Empty string for a single (non-compound) question — nothing to add to
    the prompt. For a compound question, an explicit numbered breakdown so
    the model addresses each part, in the order rule 5 requires (already
    the order `classified` is in — see _retrieve_for_question).
    """
    if len(classified) < 2:
        return ""
    lines = []
    for i, item in enumerate(classified, start=1):
        tier_label = "background/definitional" if item["type"] == "definitional" else "study/evidence"
        lines.append(f"{i}. ({tier_label}) {item['question']}")
    return (
        "This question was split into the following parts — answer each, in this order:\n"
        + "\n".join(lines)
        + "\n\n"
    )


def retrieved_population_set(chunks: List[RetrievedChunk]) -> Set[str]:
    populations = {(chunk.metadata or {}).get("population", "") for chunk in chunks}
    populations.discard("")
    return populations


def retrieved_populations_summary(chunks: List[RetrievedChunk]) -> str:
    populations = retrieved_population_set(chunks)
    return ", ".join(sorted(populations)) if populations else "unknown"


def is_population_mismatch(requested_population: Set[str], retrieved: Set[str]) -> bool:
    """
    Deterministic mismatch flag (independent of what the model writes in
    prose) so the frontend can render the warning banner reliably.

    `requested_population` is a *set* (classify_population can now name more
    than one population, e.g. {"type1", "type2"} for "What is type 1 and
    type 2 diabetes" — collapsing that to "mixed" and then comparing a
    single string was the actual bug: "type 1 and type 2 diabetes" used to
    get silently classified as type2-only, which made a mismatch fire
    against evidence that in fact covered it). A mismatch only fires when
    the question names a population the evidence doesn't cover at all —
    "mixed" retrieved content (general/all-population material) counts as
    covering everything asked, so it never triggers a mismatch either.
    """
    if not requested_population or requested_population == {"mixed"} or not retrieved:
        return False
    if "mixed" in retrieved:
        return False
    return not (requested_population & retrieved)


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


def _score_distribution_payload(decision: RetrievalDecision, namespace: str) -> Dict[str, Any]:
    return {
        "namespace": namespace,
        "floor": decision.floor,
        "background_floor": decision.background_floor if decision.background_floor is not None else decision.floor,
        "low_confidence_margin": decision.low_confidence_margin,
        "candidate_scores": decision.candidate_scores,
        "surviving_count": len(decision.surviving_chunks),
        "top_score": decision.top_score,
    }


def _is_evidence_chunk(chunk: RetrievedChunk) -> bool:
    return (chunk.metadata or {}).get("source_type", "evidence") == "evidence"


def _evidence_confidence_state(evidence_chunks: List[RetrievedChunk], floor: float, low_confidence_margin: float) -> str:
    """
    answered vs. answered_low_confidence, based on the top *evidence-tier*
    score specifically — a background chunk with a high embedding score
    shouldn't make a research-backed answer look more confident than it is.
    """
    if evidence_chunks[0].score < floor + low_confidence_margin:
        return "answered_low_confidence"
    return "answered"


def determine_final_state(chunks: List[RetrievedChunk], sentences: List[Dict[str, Any]], decision: RetrievalDecision) -> str:
    """
    The authoritative state — computed only after generation, since it
    depends on which chunks the model actually cited, not just which
    cleared the floor. An "evidence" chunk can clear the floor and still
    go unused if the model (correctly) decides it doesn't actually answer
    the question; that case is "no_evidence_for_claim", not "answered".
    """
    evidence_indices = {i for i, c in enumerate(chunks, start=1) if _is_evidence_chunk(c)}
    evidence_backed = any(evidence_indices.intersection(s.get("chunk_ids") or []) for s in sentences)

    if not evidence_backed:
        return "no_evidence_for_claim"

    evidence_chunks = [c for c in chunks if _is_evidence_chunk(c)]
    return _evidence_confidence_state(evidence_chunks, decision.floor, decision.low_confidence_margin)


def _provisional_state(decision: RetrievalDecision) -> str:
    """
    A cheap pre-generation guess for the "sources" event, sent before the
    model has produced anything — usually right, occasionally corrected by
    the authoritative state in "done" (e.g. the model declines to use an
    evidence chunk that technically cleared the floor).
    """
    evidence_chunks = [c for c in decision.surviving_chunks if _is_evidence_chunk(c)]
    if not evidence_chunks:
        return "no_evidence_for_claim"
    return _evidence_confidence_state(evidence_chunks, decision.floor, decision.low_confidence_margin)


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

    Any other stray text (leftover "<<<CONTRADICTIONS>>>" marker text from
    the non-compliance fallback, stray prose, whatever) is skipped over
    rather than treated as a reason to give up — this scanner used to bail
    out permanently on the first unexpected character, which meant a long
    contradiction analysis that blew past MAX_PREAMBLE_BUFFER before
    closing could poison the rest of the stream and silently produce zero
    sentences. Skipping forward to the next "{" is safe: anything that
    isn't actually a {sentence, chunk_ids, supported} object gets filtered
    out by _validate_sentence() downstream regardless.

    An incomplete trailing object (still streaming in) is left for the
    next call; a buffer with no "{" at all simply yields no objects yet,
    without raising, and resumes correctly once more text arrives.
    """
    i, n = start, len(buffer)
    found = []
    while i < n:
        c = buffer[i]
        if c in " \t\r\n,[]":
            i += 1
            continue
        if c != "{":
            next_brace = buffer.find("{", i)
            if next_brace == -1:
                i = n  # nothing found yet in what's streamed so far
                break
            i = next_brace
            continue

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
    Coerce a parsed JSON object into the {sentence, chunk_ids, supported,
    new_paragraph} shape, or return None if it's missing the one thing
    that actually matters (sentence text) — never propagate a malformed
    entry. new_paragraph defaults to False (stay in the current paragraph)
    if the model omits it or sends something that isn't a bool — a model
    that doesn't comply with this field just renders as one paragraph,
    same as before this field existed, rather than breaking.
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
    new_paragraph = obj.get("new_paragraph")
    if not isinstance(new_paragraph, bool):
        new_paragraph = False
    return {
        "sentence": sentence.strip(),
        "chunk_ids": chunk_ids,
        "supported": supported,
        "new_paragraph": new_paragraph,
    }


def _source_type_for_indices(indices: List[Any], chunks: List[RetrievedChunk]) -> Optional[str]:
    """
    The tier backing a sentence's citations. "evidence" wins if any cited
    chunk is evidence-tier (rule 2 forbids a sentence from mixing tiers for
    a research claim, but this stays defensive rather than assuming
    compliance); "background" only if every cited chunk is background;
    None if chunk_ids is empty or resolves to nothing valid.
    """
    types = set()
    for idx in indices:
        if isinstance(idx, int) and 1 <= idx <= len(chunks):
            types.add("evidence" if _is_evidence_chunk(chunks[idx - 1]) else "background")
    if "evidence" in types:
        return "evidence"
    if "background" in types:
        return "background"
    return None


def _sentence_payload(sentence: Dict[str, Any], chunks: List[RetrievedChunk]) -> Dict[str, Any]:
    return {
        "sentence": sentence["sentence"],
        "chunk_ids": sentence["chunk_ids"],
        "pmids": _pmids_for_indices(sentence["chunk_ids"], chunks),
        "supported": sentence["supported"],
        "source_type": _source_type_for_indices(sentence["chunk_ids"], chunks),
        "new_paragraph": sentence.get("new_paragraph", False),
    }


def compute_groundedness(sentences: List[Dict[str, Any]], chunks: List[RetrievedChunk]) -> Dict[str, Any]:
    """
    The headline metric: share of sentences with at least one supporting
    chunk, split by tier. A background-only answer looking "grounded"
    overall would hide that no research literature actually backs it, so
    groundedness_evidence is the number that matters for "is this claim
    backed by a study" — groundedness_background is definitional coverage
    only, per rule 2.
    """
    total = len(sentences)
    evidence_supported = 0
    background_supported = 0
    for s in sentences:
        source_type = _source_type_for_indices(s.get("chunk_ids") or [], chunks)
        if source_type == "evidence":
            evidence_supported += 1
        elif source_type == "background":
            background_supported += 1
    supported = evidence_supported + background_supported
    return {
        "groundedness": round(supported / total, 4) if total else None,
        "groundedness_evidence": round(evidence_supported / total, 4) if total else None,
        "groundedness_background": round(background_supported / total, 4) if total else None,
        "supported_sentences": supported,
        "evidence_supported_sentences": evidence_supported,
        "background_supported_sentences": background_supported,
        "total_sentences": total,
    }


def stream_answer(
    question: str,
    namespace: Optional[str] = None,
    *,
    retrieval_mode: str = "topk",
    force_judge: Optional[bool] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Run the grounded RAG pipeline for one question, yielding a sequence of
    event dicts: {"event": ..., "data": ...}. Possible events:

    - "sources": retrieved chunks + population info + a *provisional*
      state guess + score distribution + the judge's verdict, sent once,
      before generation. question_type ("definitional"/"evidence_seeking"/
      "compound") and sub_questions (the classify_and_decompose()
      breakdown — a list of one if the question wasn't compound) show how
      the question was routed: a definitional sub-question retrieves from
      the background tier only, an evidence-seeking one from the study
      tier only (see _retrieve_for_question). requested_population is now
      a list (can name more than one population, e.g. ["type1", "type2"]).
    - "contradictions": conflicting findings detected across excerpts (an
      empty list if none), sent once, before any "sentence" events
    - "sentence": one {sentence, chunk_ids, pmids, supported, source_type,
      new_paragraph} object, sent as soon as it completes in the stream —
      source_type is "evidence"/"background"/None, computed from chunk_ids
      so the frontend and eval harness never need to re-derive it from the
      sources list themselves; new_paragraph signals a paragraph break
      (see SYSTEM_PROMPT rule 3)
    - "done": generation finished — carries the *authoritative* state, the
      disclaimer, and the groundedness summary (share of sentences with a
      supporting chunk)
    - "refusal": either nothing was retrieved at all (state="out_of_scope",
      a defensive fallback — see below) or the answerability judge decided
      the retrieved excerpts don't actually answer the question
      (state="out_of_scope", judge.answerable=false) — refusal message,
      closest scores found, what the system covers, the score
      distribution, and the judge's verdict+reason. No main-model call is
      made either way (the judge call, on a smaller model, still runs).
    - "error": something is misconfigured (e.g. missing API key) or
      generation itself failed

    The API's state enum is answered / answered_low_confidence /
    no_evidence_for_claim / out_of_scope. "out_of_scope" is now decided by
    src.rag.judge.judge_answerability, not a similarity threshold —
    docs/findings-retrieval-floor.md found cosine similarity measures
    topical relatedness, not answerability, so it can't separate a
    diabetes-adjacent-but-out-of-scope question from a genuinely in-scope
    one. The similarity floor is still computed and logged (see
    _retrieve_for_question) and still used for score_distribution in the
    inspector, but no longer gates generation on its own; the one place it
    still can is the defensive "nothing retrieved at all" case handled
    just below (e.g. an empty namespace), which the judge is never even
    asked about. Beyond the judge's gate, the remaining three states
    depend on which chunks the model actually cites — a chunk can pass the
    judge and still go unused if the model correctly decides it doesn't
    answer the question, which is exactly "no_evidence_for_claim": in
    scope, and (when background chunks exist) the model still gives the
    definitional context, but no study addresses the specific claim
    asked. That's why "sources" carries a provisional guess (usually
    right, cheap, lets the frontend show something immediately) while
    "done" carries the real answer, computed by determine_final_state()
    from what was actually cited.

    Retrieves a wide candidate pool (RETRIEVAL_CANDIDATE_K) per
    sub-question for scoring, but only the top get_retriever_k() by raw
    score are used for the judge/generation. Never sends more than
    RETRIEVAL_CANDIDATE_K chunks to the main model — enforced by a hard
    assertion, not just a printed warning.

    retrieval_mode and force_judge are diagnostic-only, keyword-only
    parameters for eval scripts reconstructing the pre-judge pipeline for
    comparison (see scripts/run_old_pipeline_full.py) — the live /query
    path never passes them, so production behavior is unchanged by their
    existence. retrieval_mode="floor" selects only floor-surviving chunks
    (see _retrieve_for_question) instead of the top-k-by-raw-score default.
    force_judge, when not None, skips the real judge call entirely and
    uses this boolean as the verdict; the emitted judge payload still
    carries {"answerable": force_judge, "bypassed": True, "reason": ...}
    so any record built from these events is visibly marked as using a
    forced verdict, never mistaken for a real judge call.
    """
    t_start = time.monotonic()

    try:
        client = get_client()
    except ValueError as e:
        yield {"event": "error", "data": {"message": str(e)}}
        return

    requested_population = classify_population(question)
    requested_population_str = ", ".join(sorted(requested_population))
    resolved_namespace = resolve_namespace(namespace)

    try:
        decision, classified_sub_questions = _retrieve_for_question(question, namespace, retrieval_mode=retrieval_mode)
    except ValueError as e:
        yield {"event": "error", "data": {"message": str(e)}}
        return

    question_type = overall_question_type(classified_sub_questions)

    t_after_retrieval = time.monotonic()
    retrieval_ms = round((t_after_retrieval - t_start) * 1000, 1)

    assert_chunk_cap(decision.surviving_chunks, RETRIEVAL_CANDIDATE_K)

    if decision.state == "out_of_scope":
        # Defensive fallback only — fires when nothing was retrieved at all
        # (e.g. an empty namespace), not as the answerability gate: that's
        # the judge's job now, below. Never asks the judge about a question
        # with literally nothing to show it.
        closest_scores = decision.candidate_scores[:3]
        log_query_event(question, sorted(requested_population), [], OUT_OF_SCOPE_TEXT, state="out_of_scope")
        yield {
            "event": "refusal",
            "data": {
                "state": "out_of_scope",
                "message": OUT_OF_SCOPE_TEXT,
                "scope_description": SCOPE_DESCRIPTION,
                "closest_scores": closest_scores,
                "score_distribution": _score_distribution_payload(decision, resolved_namespace),
                "judge": None,
                "disclaimer": DISCLAIMER,
                "timing_ms": {"retrieval": retrieval_ms},
            },
        }
        return

    chunks = decision.surviving_chunks

    t_before_judge = time.monotonic()
    if force_judge is not None:
        judge_result = {
            "answerable": force_judge,
            "reason": "judge bypassed (diagnostic run) — verdict forced by caller, not a real judge call",
            "bypassed": True,
        }
    else:
        judge_result = judge_answerability(client, question, chunks[:JUDGE_CHUNK_LIMIT])
    judge_ms = round((time.monotonic() - t_before_judge) * 1000, 1)

    if not judge_result["answerable"]:
        log_query_event(question, sorted(requested_population), [], JUDGE_REFUSAL_TEXT, state="out_of_scope")
        yield {
            "event": "refusal",
            "data": {
                "state": "out_of_scope",
                "message": JUDGE_REFUSAL_TEXT,
                "scope_description": SCOPE_DESCRIPTION,
                "closest_scores": decision.candidate_scores[:3],
                "score_distribution": _score_distribution_payload(decision, resolved_namespace),
                "judge": judge_result,
                "disclaimer": DISCLAIMER,
                "timing_ms": {"retrieval": retrieval_ms, "judge": judge_ms},
            },
        }
        return

    retrieved = retrieved_population_set(chunks)
    mismatch = is_population_mismatch(requested_population, retrieved)

    yield {
        "event": "sources",
        "data": {
            "state": _provisional_state(decision),
            "question_type": question_type,
            "sub_questions": classified_sub_questions,
            "requested_population": sorted(requested_population),
            "retrieved_populations": sorted(retrieved),
            "mismatch": mismatch,
            "score_distribution": _score_distribution_payload(decision, resolved_namespace),
            "judge": judge_result,
            "sources": [_source_payload(c) for c in chunks],
            "timing_ms": {"retrieval": retrieval_ms, "judge": judge_ms},
        },
    }

    context_str = format_context(chunks)
    retrieved_populations_str = retrieved_populations_summary(chunks)
    system_prompt = SYSTEM_PROMPT.format(
        requested_population=requested_population_str,
        retrieved_populations=retrieved_populations_str,
        sub_question_note=_sub_question_note(classified_sub_questions),
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
            max_tokens=8192,
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
                    # Model didn't emit the expected format — stop waiting
                    # and treat everything buffered so far as the start of
                    # its answer content.
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
            # Stream ended entirely within the buffering window (a very
            # short response).
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
            # Nothing parsed as a valid sentence object at all — the model
            # didn't comply with the JSON format. Rare, but the API
            # contract always returns at least one sentence object.
            leftover = sentence_buffer.strip()
            if leftover:
                synthetic = {"sentence": leftover, "chunk_ids": [], "supported": False}
                full_sentences.append(synthetic)
                yield {"event": "sentence", "data": _sentence_payload(synthetic, chunks)}

        # Real usage from the API (not the rough char/4 estimate used for
        # the pre-call safety check) — the actual figure for cost/latency
        # reporting. Also check stop_reason: a response cut off by
        # max_tokens must never be silently reported as the model having
        # concluded "no evidence for this claim" — those are very
        # different situations (a token budget ran out vs. a genuine
        # absence of supporting literature) and conflating them would
        # misreport real generation failures as an honest evidence gap.
        usage = None
        truncated = False
        model_used = None
        try:
            final_message = stream.get_final_message()
            usage = {
                "input_tokens": final_message.usage.input_tokens,
                "output_tokens": final_message.usage.output_tokens,
            }
            truncated = final_message.stop_reason == "max_tokens"
            # The exact model string the API actually served this request
            # with, not get_model()'s config value — a report on "which
            # model was actually called" should read this field, not the
            # config default, since a model alias can resolve to a
            # different pinned snapshot than the name requested.
            model_used = final_message.model
        except Exception:
            pass  # usage is a nice-to-have; never fail the response over it
    except anthropic.APIError as e:
        # A partial answer may already have streamed — surface the failure
        # explicitly rather than letting the connection die silently.
        yield {"event": "error", "data": {"message": f"Generation failed: {e}"}}
        return

    full_answer = " ".join(s["sentence"] for s in full_sentences)
    groundedness = compute_groundedness(full_sentences, chunks)
    final_state = determine_final_state(chunks, full_sentences, decision)
    t_end = time.monotonic()
    timing_ms = {
        "retrieval": retrieval_ms,
        "judge": judge_ms,
        "generation": round((t_end - t_after_retrieval) * 1000 - judge_ms, 1),
        "total": round((t_end - t_start) * 1000, 1),
    }

    log_query_event(
        question, sorted(requested_population), chunks, full_answer,
        state=final_state, sentences=full_sentences, groundedness=groundedness["groundedness"],
    )

    yield {
        "event": "done",
        "data": {
            "state": final_state,
            "disclaimer": DISCLAIMER,
            "groundedness": groundedness,
            "timing_ms": timing_ms,
            "usage": usage,
            "truncated": truncated,
            "model": model_used,
        },
    }
