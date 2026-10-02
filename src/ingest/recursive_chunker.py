"""
v3 recursive chunker (namespace v3_5k_recursive).

Split order, coarsest first: the whole abstract -> abstract sections -> sentences
-> words (last resort, only for a single sentence too long to fit on its own).
Pieces are packed greedily up to the limit, so a chunk never cuts mid-sentence
unless one sentence alone exceeds it.

The limit is in MiniLM tokens, counted on exactly the text that gets embedded:
the header line plus the chunk body must come to at most MAX_CHUNK_TOKENS, which
stays under the embedder's 256-token truncation (src.ingest.local_embeddings).

Overlap: the last full sentence of a chunk is carried forward as the first
sentence of the next chunk of the same document, when it fits.

Header on every chunk: "Title | Year | Population | Study design".

The old splitters in src.ingest.chunker are untouched; v1_300 and v2_5k were
built with them and are never re-embedded.
"""
import re
from typing import Callable, Dict, List, Optional

from src.ingest.local_embeddings import count_tokens

MAX_CHUNK_TOKENS = 250

POPULATION_LABELS = {
    "type1": "Type 1 diabetes",
    "type2": "Type 2 diabetes",
    "gestational": "Gestational diabetes",
    "prediabetes": "Prediabetes",
    "mixed": "Mixed or general diabetes",
}

RETRACTED_TYPE = "Retracted Publication"

# Abbreviations that end in a period but don't end a sentence.
_ABBREVIATIONS = (
    "e.g.", "i.e.", "vs.", "et al.", "approx.", "etc.", "Fig.", "fig.", "No.", "no.",
    "Dr.", "Mr.", "Ms.", "Prof.", "St.", "cf.", "ca.", "resp.", "incl.", "U.S.",
)
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]|[A-Z0-9])")


def study_design(publication_type: str) -> str:
    """
    Evidence badge from a PubMed publication-type string such as
    "Journal Article; Systematic Review; Meta-Analysis". Strongest design wins.
    "Clinical trial" covers trials PubMed doesn't tag as randomized (phase I-IV,
    non-randomized, pragmatic), which would be mislabeled as either RCT or
    observational.
    """
    types = {t.strip().lower() for t in (publication_type or "").split(";")}
    if any("meta-analysis" in t for t in types):
        return "Meta-analysis"
    if "randomized controlled trial" in types or "equivalence trial" in types:
        return "RCT"
    if types & {"observational study", "case reports", "comparative study", "validation study"} and not any(
        "review" in t or "trial" in t for t in types
    ):
        return "Observational"
    if any("review" in t for t in types) or types & {"consensus statement", "practice guideline", "guideline"}:
        return "Review"
    if any("trial" in t for t in types):
        return "Clinical trial"
    return "Observational"


def is_retracted(publication_type: str) -> bool:
    return RETRACTED_TYPE.lower() in (publication_type or "").lower()


def split_sentences(text: str) -> List[str]:
    """Sentence split that doesn't break on common abbreviations or decimals."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    placeholder = "\u0000"
    protected = text
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", placeholder))
    parts = _SENTENCE_END.split(protected)
    return [p.replace(placeholder, ".").strip() for p in parts if p.strip()]


def make_header(title: str, year, population: str, design: str) -> str:
    pop = POPULATION_LABELS.get(population or "", population or "Unknown population")
    return f"{(title or '').strip()} | {year or 'n.d.'} | {pop} | {design}"


def _split_words(sentence: str, fits: Callable[[str], bool]) -> List[str]:
    """Last resort: greedy word packing for one sentence too long to fit alone."""
    pieces, current = [], []
    for word in sentence.split():
        candidate = " ".join(current + [word])
        if current and not fits(candidate):
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_document_text(
    header: str,
    sections: List[Dict[str, str]],
    max_tokens: int = MAX_CHUNK_TOKENS,
    token_counter: Callable[[str], int] = count_tokens,
) -> List[Dict]:
    """
    Recursively split one document's sections into chunk bodies.

    Returns [{"body": str, "sections": [labels]}] in document order. The text
    embedded for a chunk is f"{header}\\n{body}", and that string is what is
    measured against max_tokens.
    """
    def fits(body: str) -> bool:
        return token_counter(f"{header}\n{body}") <= max_tokens

    def labelled(label: str, text: str) -> str:
        return f"{label}: {text}" if label else text

    sections = [s for s in sections if (s.get("text") or "").strip()]
    if not sections:
        return []

    whole = " ".join(labelled(s.get("label", ""), s["text"].strip()) for s in sections)
    if fits(whole):
        return [{"body": whole, "sections": [s.get("label", "") for s in sections]}]

    # Level 2: pack whole sections. A section that fits alone is never split.
    # Level 3/4: a section that doesn't fit is broken into sentences (and a
    # too-long sentence into words), each unit tagged with its section label.
    units = []  # (text, label, is_full_sentence)
    for s in sections:
        label, text = s.get("label", ""), s["text"].strip()
        section_text = labelled(label, text)
        if fits(section_text):
            units.append((section_text, label, False))
            continue
        sentences = split_sentences(text)
        for i, sentence in enumerate(sentences):
            sentence_text = labelled(label, sentence) if i == 0 else sentence
            if fits(sentence_text):
                units.append((sentence_text, label, True))
            else:
                for piece in _split_words(sentence_text, fits):
                    units.append((piece, label, False))

    chunks: List[Dict] = []
    current: List[str] = []
    current_labels: List[str] = []
    last_sentence: Optional[str] = None  # last full sentence of the previous chunk

    def flush():
        nonlocal current, current_labels
        if current:
            chunks.append({"body": " ".join(current), "sections": list(dict.fromkeys(current_labels))})
        current, current_labels = [], []

    for text, label, is_sentence in units:
        if current and fits(" ".join(current + [text])):
            current.append(text)
            current_labels.append(label)
            continue
        if current:
            tail = _last_full_sentence(current)
            flush()
            last_sentence = tail
        # Start a new chunk, carrying the previous chunk's last full sentence when it fits.
        if last_sentence and last_sentence != text and fits(f"{last_sentence} {text}"):
            current = [last_sentence, text]
        else:
            current = [text]
        current_labels.append(label)
        last_sentence = None
    flush()
    return chunks


def _last_full_sentence(parts: List[str]) -> Optional[str]:
    """The final complete sentence of a chunk's last unit, or None if it ends mid-sentence (word split)."""
    sentences = split_sentences(parts[-1])
    if not sentences:
        return None
    last = sentences[-1]
    if not re.search(r"[.!?][\"')\]]*$", last):
        return None
    return re.sub(r"^[A-Z][A-Z /&-]{1,40}: ", "", last)


def split_pubmed_documents_recursive(documents: List[Dict], max_tokens: int = MAX_CHUNK_TOKENS) -> List[Dict]:
    """
    Chunk PubMed abstracts for v3_5k_recursive. Skips any abstract tagged
    "Retracted Publication". Every chunk carries its parent PMID, publication
    type, study design and the header it was embedded with.
    """
    chunked = []
    for doc in documents:
        pub_type = doc.get("publication_type", "")
        if is_retracted(pub_type):
            continue
        design = study_design(pub_type)
        header = make_header(doc.get("title", ""), doc.get("year", ""), doc.get("population", ""), design)
        sections = doc.get("sections") or [{"label": "", "text": doc.get("abstract", "")}]
        for i, piece in enumerate(chunk_document_text(header, sections, max_tokens=max_tokens)):
            chunked.append({
                "text": f"{header}\n{piece['body']}",
                "metadata": {
                    "source_type": "evidence",
                    "pmid": doc.get("pmid", ""),
                    "parent_id": doc.get("pmid", ""),
                    "title": doc.get("title", ""),
                    "journal": doc.get("journal", ""),
                    "year": doc.get("year", ""),
                    "publication_type": pub_type,
                    "study_design": design,
                    "population": doc.get("population", ""),
                    "population_confidence": doc.get("population_confidence", ""),
                    "section": ", ".join(l for l in piece["sections"] if l),
                    "chunk_index": i,
                },
            })
    return chunked


def background_parent_id(publisher: str, title: str) -> str:
    slug = lambda s: re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")
    return f"bg_{slug(publisher)}_{slug(title)}"


def split_background_documents_recursive(documents: List[Dict], max_tokens: int = MAX_CHUNK_TOKENS) -> List[Dict]:
    """
    Same recursive split for the ADA/NIDDK/CDC background pages. They have no
    section labels, so the page text is the single top-level section.
    """
    chunked = []
    for doc in documents:
        design = f"Background ({doc.get('publisher', '')})"
        header = make_header(doc.get("title", ""), doc.get("year", ""), doc.get("population", ""), design)
        parent = background_parent_id(doc.get("publisher", ""), doc.get("title", ""))
        for i, piece in enumerate(chunk_document_text(header, [{"label": "", "text": doc.get("text", "")}], max_tokens=max_tokens)):
            chunked.append({
                "text": f"{header}\n{piece['body']}",
                "metadata": {
                    "source_type": "background",
                    "parent_id": parent,
                    "publisher": doc.get("publisher", ""),
                    "title": doc.get("title", ""),
                    "source_url": doc.get("source_url", ""),
                    "population": doc.get("population", ""),
                    "publication_type": "Background",
                    "study_design": design,
                    "chunk_index": i,
                },
            })
    return chunked


def parent_text(doc: Dict) -> str:
    """The full parent document text sent to the evidence judge and generation."""
    if doc.get("sections"):
        return " ".join(
            f"{s['label']}: {s['text'].strip()}" if s.get("label") else s["text"].strip()
            for s in doc["sections"] if (s.get("text") or "").strip()
        )
    return (doc.get("abstract") or doc.get("text") or "").strip()
