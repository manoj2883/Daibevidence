"""
v3 answer pipeline, plus the dispatcher the API calls.

SCOPE_JUDGE_ENABLED (env, default false) selects between:
- false (default): this module's pipeline —
    1. query step (src.rag.query_rewriter, one Haiku call) -> search_query, keywords, personal_dosing
    2. personal_dosing -> status "see_clinician", nothing retrieved or generated
    3. hybrid retrieval (src.rag.hybrid_retriever) over PubMed chunks AND background pages,
       expanded to full parent documents
    4. evidence judge (src.rag.evidence_judge, unchanged prompt, same in both audiences)
    5. generation for the chosen audience ("patient" default, or "clinician")
- true: the retired two-stage pipeline, src.rag.chain.stream_answer (kept, unchanged).

Statuses: answered | answered_partial | not_covered | see_clinician.

The query step's output is internal. The evidence judge and the generation prompt
always receive the user's original question; search_query/keywords/personal_dosing
appear only in the "inspector" payload, which the UI shows in the retrieval inspector.
"""
import os
import time
from typing import Any, Dict, Generator, List, Optional

from src.ingest.config import NAMESPACE_V3
from src.ingest.population import classify_population
from src.rag.chain import (
    CONTRA_END,
    CONTRA_START,
    DISCLAIMER,
    _evidence_audit_note,
    _status_instruction,
    compute_groundedness,
    filter_chunks_to_direct,
    get_client,
    is_population_mismatch,
    retrieved_population_set,
    retrieved_populations_summary,
    stream_answer as stream_answer_two_stage,
    stream_generation,
)
from src.rag.cost import print_prompt_estimate
from src.rag.evidence_judge import judge_evidence
from src.rag.hybrid_retriever import dense_chunks_retrieve, hybrid_retrieve
from src.rag.query_log import log_query_event
from src.rag.query_rewriter import rewrite_query
from src.rag.retriever import resolve_namespace
from src.rag.safety import MEDICATION_NOTE, touches_medication
from src.rag.types import RetrievedChunk

AUDIENCES = ("patient", "clinician")
RETRIEVAL_STRATEGIES = ("hybrid", "dense_chunks")

NOT_COVERED_TEXT = "Our research library doesn't cover this question yet."
SEE_CLINICIAN_TEXT = (
    "This question asks about a dose or a medication change for you personally. That decision needs "
    "someone who knows your health history, so please ask your doctor or pharmacist."
)


def scope_judge_enabled() -> bool:
    return os.environ.get("SCOPE_JUDGE_ENABLED", "false").strip().lower() in ("1", "true", "yes")


PATIENT_RULES = """AUDIENCE: a patient or family member, not a health professional.
- Write in plain English with short sentences (under 20 words where you can).
- The first time you use a medical term, explain it in plain words in the same sentence, e.g. \
"HbA1c (a blood test that shows your average blood sugar over about three months)".
- NEVER name a specific drug, drug class, or supplement unless the user's question names it, even \
when the excerpts name it. Say "some diabetes medicines" (or "some supplements") instead.
- Describe results in plain words ("lowered average blood sugar a little"). Use a simple number only \
when it helps; leave out confidence intervals, p-values, and statistical terms.
- Do not mention PMIDs, journal names, or excerpt numbers in your sentences.
- Do not add any closing advice about doctors or pharmacists; the system adds that itself."""

CLINICIAN_RULES = """AUDIENCE: a clinician.
- Put the study details up front in each evidence sentence: design, population, size, and PMID, \
e.g. "A 2021 meta-analysis of 14 RCTs in adults with type 2 diabetes (PMID 12345678) found ...".
- Report effect sizes exactly as the excerpts state them (mean differences, CIs, p-values, follow-up).
- Use precise drug names and drug classes as the excerpts name them.
- Technical vocabulary is fine; stay concise."""

SYSTEM_PROMPT_V3 = """You are a clinical evidence assistant covering all forms of diabetes \
(type 1, type 2, gestational, and prediabetes) research. You answer strictly from the source \
documents below. Each document is labeled SOURCE TYPE in its header: "background" (patient-education \
pages from ADA, NIDDK or CDC, covering definitions, diagnostic criteria, and general disease mechanism \
— not research findings) or "study" (a full PubMed abstract: a review, systematic review, \
meta-analysis, or clinical trial). You are not a doctor and nothing you say is medical advice.

{audience_rules}

Rules, in order of priority:
1. Use ONLY information stated in the source documents. Never use outside knowledge, training data, \
or assumptions, even if you believe you know the answer.
2. A "background" document may ONLY support a sentence that defines or explains something — what a \
condition is, how it's diagnosed, or how it generally works. It must NEVER be the sole or partial \
support for a sentence that claims an intervention works, states or implies an effect size, makes a \
comparison, or reports any research finding — that requires a "study" document. If a sentence mixes a \
definitional point with a research claim, split it into separate sentences.
3. Output your answer as a single JSON array, one object per sentence, in order. Each object has \
exactly these keys: "sentence" (one complete sentence, plain text, {pmid_rule}), "chunk_ids" (array of \
the [Excerpt N] numbers that support this specific sentence, e.g. [1, 3], or [] if the sentence is \
pure framing with no factual claim), "supported" (true only if chunk_ids is non-empty and the sentence \
genuinely follows from those documents, respecting rule 2), and "new_paragraph" (true if this sentence \
starts a new paragraph). Keep paragraphs short; start a new paragraph at each shift in idea, and \
always between background content and study content. If a specific claim cannot be attributed to any \
document, leave it out entirely.
4. Your output MUST begin with the contradiction block, before the sentence array. Compare the "study" documents for genuine contradictions — two documents \
reporting conflicting findings on the same specific claim (not different topics, and not a population \
difference). Report this as a line containing exactly "{contra_start}", then a JSON array (or "[]" if \
none), then a line containing exactly "{contra_end}". Each entry has exactly these keys: \
"excerpt_indices", "claim", "position_a", "position_b", "differs_by". If you report a contradiction, \
your sentence array MUST present both positions.
5. Lead with the direct answer — the first sentence answers the question, never a caveat. When the \
question has a definitional part and a research part, answer the definitional part first from \
background documents, then what the studies show. State which diabetes population (type 1, type 2, \
gestational, prediabetes, or a mix) the evidence applies to, as one of your sentences. Every caveat \
comes AFTER the direct answer.
6. The question was automatically classified as being about population(s) "{requested_population}" \
("mixed" means none named). The documents cover population(s): {retrieved_populations}. Flag a \
population mismatch, as a caveat after your main answer, only when "{requested_population}" is not \
"mixed" AND none of the named population(s) are among the documents' populations AND those are not \
"mixed". Otherwise just state which population(s) the evidence covers.
7. If no document answers a specific claim in the question, end with exactly one plain-language \
sentence saying the studies found here don't address [the specific claim], with chunk_ids [] and \
supported false. Never stretch a document to appear to answer something it doesn't.
8. Never fabricate a PMID, a statistic, a study finding, or a population.
9. If the question or the evidence concerns diabetes medications, discuss them only in general, \
informational terms. NEVER give dosing instructions, prescribing guidance, or advice to start, stop, \
or adjust a medication.
10. {status_instruction}

{evidence_audit_note}Source documents:
{context}"""

PMID_RULE = {
    "patient": "never containing PMIDs or bracketed citations — attribution is carried by chunk_ids",
    "clinician": "naming the study's PMID in parentheses where it reports a finding, e.g. (PMID 12345678); "
                 "never use bracketed citations like [1]",
}


def format_context_v3(chunks: List[RetrievedChunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata or {}
        population = meta.get("population", "")
        if meta.get("source_type") == "background":
            header = (f"[Excerpt {i}] SOURCE TYPE: background | Publisher: {meta.get('publisher', '')} — "
                      f"{meta.get('title', '')} | Population: {population}")
        else:
            header = (f"[Excerpt {i}] SOURCE TYPE: study | PMID {meta.get('pmid', 'unknown')} — {meta.get('title', '')} "
                      f"({meta.get('journal', '')}, {meta.get('year', '')}) | Study design: {meta.get('study_design', '')} "
                      f"| Population: {population}")
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def evidence_badge(meta: Dict[str, Any]) -> str:
    if meta.get("source_type") == "background":
        return f"Background ({meta.get('publisher', '')})"
    if meta.get("study_design"):
        return meta["study_design"]
    from src.ingest.recursive_chunker import study_design
    return study_design(meta.get("publication_type", ""))


def source_payload_v3(chunk: RetrievedChunk) -> Dict[str, Any]:
    meta = chunk.metadata or {}
    return {
        "source_type": meta.get("source_type", "evidence"),
        "pmid": meta.get("pmid", ""),
        "title": meta.get("title", ""),
        "journal": meta.get("journal", ""),
        "year": meta.get("year", ""),
        "publication_type": meta.get("publication_type", ""),
        "evidence_badge": evidence_badge(meta),
        "publisher": meta.get("publisher", ""),
        "source_url": meta.get("source_url", ""),
        "population": meta.get("population", ""),
        "excerpt": chunk.text,
        "matched_chunk": meta.get("matched_chunk_text", ""),
        "fusion_rank": meta.get("fusion_rank"),
        "dense_rank": meta.get("dense_rank"),
        "bm25_rank": meta.get("bm25_rank"),
        "score": chunk.score,
    }


def _resolve_v3_namespace(namespace: Optional[str], strategy: str) -> str:
    if namespace is not None:
        return namespace
    return os.environ.get("PINECONE_NAMESPACE", NAMESPACE_V3) if strategy == "hybrid" else resolve_namespace(None)


def stream_answer_v3(
    question: str,
    namespace: Optional[str] = None,
    *,
    audience: str = "patient",
    retrieval_strategy: str = "hybrid",
) -> Generator[Dict[str, Any], None, None]:
    """
    Events: "query" (inspector only) -> "sources" -> "contradictions" -> "sentence"* -> "done",
    or "query" -> "refusal" (not_covered / see_clinician), or "error".
    retrieval_strategy="dense_chunks" is the eval's Run A (v2_5k + background, no rewrite/BM25/parents).
    """
    if audience not in AUDIENCES:
        raise ValueError(f"audience must be one of {AUDIENCES}, got {audience!r}")
    if retrieval_strategy not in RETRIEVAL_STRATEGIES:
        raise ValueError(f"retrieval_strategy must be one of {RETRIEVAL_STRATEGIES}, got {retrieval_strategy!r}")

    t_start = time.monotonic()
    try:
        client = get_client()
    except ValueError as e:
        yield {"event": "error", "data": {"message": str(e)}}
        return

    namespace = _resolve_v3_namespace(namespace, retrieval_strategy)
    requested_population = classify_population(question)

    # --- 1. query step (internal) -------------------------------------------------------------
    t0 = time.monotonic()
    query = rewrite_query(client, question)
    query_ms = round((time.monotonic() - t0) * 1000, 1)
    query_inspector = {k: query.get(k) for k in ("search_query", "keywords", "personal_dosing", "fallback", "reason", "model")}
    yield {"event": "query", "data": {"query_step": query_inspector, "audience": audience, "namespace": namespace}}

    base = {"audience": audience, "query_step": query_inspector, "disclaimer": DISCLAIMER, "personal_dosing": bool(query.get("personal_dosing"))}

    # --- 2. personal dosing -> see_clinician --------------------------------------------------
    if query.get("personal_dosing"):
        log_query_event(question, sorted(requested_population), [], SEE_CLINICIAN_TEXT, state="see_clinician")
        yield {"event": "refusal", "data": {**base, "status": "see_clinician", "state": "see_clinician",
                                            "message": SEE_CLINICIAN_TEXT, "evidence_verdict": None,
                                            "evidence_reason": None, "propositions": [], "retrieval": None,
                                            "timing_ms": {"query": query_ms}}}
        return

    # --- 3. retrieval ---------------------------------------------------------------------------
    t0 = time.monotonic()
    try:
        if retrieval_strategy == "hybrid":
            chunks, retrieval_info = hybrid_retrieve(query["search_query"], query.get("keywords") or [], namespace=namespace)
        else:
            chunks, retrieval_info = dense_chunks_retrieve(question, namespace)
    except (ValueError, FileNotFoundError) as e:
        yield {"event": "error", "data": {"message": str(e)}}
        return
    retrieval_ms = round((time.monotonic() - t0) * 1000, 1)
    retrieval_info["strategy"] = retrieval_strategy

    # --- 4. evidence judge (original question, identical across audiences) ---------------------
    t0 = time.monotonic()
    evidence = judge_evidence(client, question, chunks)
    evidence_ms = round((time.monotonic() - t0) * 1000, 1)
    verdict = evidence["verdict"]

    judge_fields = {
        "evidence_verdict": verdict,
        "evidence_reason": evidence["reason"],
        "propositions": evidence["propositions"],
        "population_mismatch": evidence["population_mismatch"],
        "question_population": evidence["question_population"],
        "evidence_populations": evidence["evidence_populations"],
        "on_topic": evidence.get("on_topic"),
        "cross_study_comparison": bool(evidence.get("cross_study_comparison")),
        "judge": evidence,
        "retrieval": retrieval_info,
    }
    timing = {"query": query_ms, "retrieval": retrieval_ms, "evidence": evidence_ms}

    if verdict == "sufficient":
        status, gen_chunks = "answered", chunks
    elif verdict == "partial":
        status, gen_chunks = "answered_partial", filter_chunks_to_direct(chunks, evidence)
    else:
        log_query_event(question, sorted(requested_population), [], NOT_COVERED_TEXT, state="not_covered")
        yield {"event": "refusal", "data": {**base, **judge_fields, "status": "not_covered", "state": "not_covered",
                                            "message": NOT_COVERED_TEXT,
                                            "considered_sources": [source_payload_v3(c) for c in chunks],
                                            "timing_ms": timing}}
        return

    retrieved = retrieved_population_set(gen_chunks)
    yield {"event": "sources", "data": {
        **base, **judge_fields, "status": status, "state": status,
        "requested_population": sorted(requested_population),
        "retrieved_populations": sorted(retrieved),
        "mismatch": is_population_mismatch(requested_population, retrieved),
        "sources": [source_payload_v3(c) for c in gen_chunks],
        "timing_ms": timing,
    }}

    # --- 5. generation ----------------------------------------------------------------------------
    system_prompt = SYSTEM_PROMPT_V3.format(
        audience_rules=PATIENT_RULES if audience == "patient" else CLINICIAN_RULES,
        pmid_rule=PMID_RULE[audience],
        requested_population=", ".join(sorted(requested_population)),
        retrieved_populations=retrieved_populations_summary(gen_chunks),
        status_instruction=_status_instruction(status, evidence),
        evidence_audit_note=_evidence_audit_note(evidence),
        contra_start=CONTRA_START,
        contra_end=CONTRA_END,
        context=format_context_v3(gen_chunks),
    )
    print_prompt_estimate(f"v3 query: {question[:60]!r}", system_prompt, question)

    t_gen = time.monotonic()
    generation: Dict[str, Any] = {}
    yield from stream_generation(client, system_prompt, question, gen_chunks, t_gen, generation)
    if generation.get("failed"):
        return
    sentences = generation["sentences"]

    medication_note = audience == "patient" and touches_medication(question, [s["sentence"] for s in sentences])
    if medication_note:
        note = {"sentence": MEDICATION_NOTE, "chunk_ids": [], "pmids": [], "supported": False,
                "source_type": None, "new_paragraph": True, "safety_note": True}
        yield {"event": "sentence", "data": note}

    groundedness = compute_groundedness(sentences, gen_chunks)
    t_end = time.monotonic()
    timing.update(generation=round((t_end - t_gen) * 1000, 1), total=round((t_end - t_start) * 1000, 1))
    full_answer = " ".join(s["sentence"] for s in sentences) + (f" {MEDICATION_NOTE}" if medication_note else "")
    log_query_event(question, sorted(requested_population), gen_chunks, full_answer, state=status,
                    sentences=sentences, groundedness=groundedness["groundedness"])

    yield {"event": "done", "data": {
        "status": status, "state": status, "audience": audience,
        "evidence_verdict": verdict, "population_mismatch": evidence["population_mismatch"],
        "medication_note": medication_note, "disclaimer": DISCLAIMER, "groundedness": groundedness,
        "timing_ms": timing, "usage": generation["usage"], "truncated": generation["truncated"],
        "model": generation["model"],
    }}


def run_pipeline(question: str, namespace: Optional[str] = None, audience: str = "patient"):
    """The API entry point: v3 by default, the two-stage pipeline when SCOPE_JUDGE_ENABLED=true."""
    if scope_judge_enabled():
        return stream_answer_two_stage(question, namespace=namespace)
    return stream_answer_v3(question, namespace=namespace, audience=audience)
