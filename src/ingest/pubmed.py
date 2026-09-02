"""
PubMed ingestion source: pulls diabetes abstracts (all types) across four
topics — diet & nutrition, glycemic control, body composition & weight, and
diet/medication interaction — from NCBI's E-utilities (esearch + efetch).
No API key required (an optional one just raises the rate limit).

Scope, filters, and limits are controlled by src.ingest.config.
"""
import csv
import json
import os
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

import requests

from src.ingest.config import (
    DATE_RANGE_YEARS,
    DIABETES_SCOPE,
    FETCH_LIMIT,
    FETCH_META_PATH,
    PUBLICATION_TYPES,
    RAW_CSV_PATH,
    RAW_JSON_PATH,
    TOPICS,
)
from src.ingest.population import classify_population

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EFETCH_BATCH_SIZE = 200

RAW_FIELDS = ["pmid", "title", "journal", "year", "publication_type", "population", "topic", "abstract"]


def _request_params(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    params: Dict[str, str] = {}
    email = os.environ.get("PUBMED_EMAIL")
    api_key = os.environ.get("PUBMED_API_KEY")
    tool = os.environ.get("PUBMED_TOOL", "diabevidence")
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    params["tool"] = tool
    if extra:
        params.update(extra)
    return params


def _throttle() -> None:
    # NCBI allows 3 req/s without an API key, 10 req/s with one.
    delay = 0.11 if os.environ.get("PUBMED_API_KEY") else 0.34
    time.sleep(delay)


def _date_range_strings(years: int = DATE_RANGE_YEARS):
    today = date.today()
    try:
        start = today.replace(year=today.year - years)
    except ValueError:
        # today is Feb 29 and (today.year - years) isn't a leap year
        start = today.replace(month=2, day=28, year=today.year - years)
    return start.strftime("%Y/%m/%d"), today.strftime("%Y/%m/%d")


def build_topic_query(topic_fragment: str) -> str:
    """
    Combine the diabetes scope, a topic fragment, the publication-type
    filter, and the humans/English/has-abstract filters into one query.
    """
    pub_type_clause = " OR ".join(f'"{pt}"[Publication Type]' for pt in PUBLICATION_TYPES)
    return (
        f"{DIABETES_SCOPE} AND {topic_fragment} "
        f"AND ({pub_type_clause}) "
        f"AND humans[MeSH Terms] AND English[lang] AND hasabstract[text]"
    )


def search_pubmed(query: str, retmax: int, mindate: str, maxdate: str) -> List[str]:
    """
    Run an esearch query restricted to [mindate, maxdate] and return matching PMIDs.
    """
    params = _request_params({
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(retmax),
        "datetype": "pdat",
        "mindate": mindate,
        "maxdate": maxdate,
    })

    resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    _throttle()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def _text(el: Optional[ET.Element]) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def _parse_article(article_el: ET.Element) -> Optional[Dict]:
    """
    Extract pmid, title, journal, year, abstract, publication_type, and a
    heuristically-classified population.
    """
    medline = article_el.find("MedlineCitation")
    if medline is None:
        return None

    pmid = _text(medline.find("PMID"))
    if not pmid:
        return None

    article = medline.find("Article")
    if article is None:
        return None

    title = _text(article.find("ArticleTitle"))

    abstract_parts = []
    abstract_el = article.find("Abstract")
    if abstract_el is not None:
        for chunk in abstract_el.findall("AbstractText"):
            label = chunk.get("Label")
            piece = _text(chunk)
            if not piece:
                continue
            abstract_parts.append(f"{label}: {piece}" if label else piece)
    abstract = "\n".join(abstract_parts)

    if not abstract:
        return None

    journal = ""
    year = ""
    journal_el = article.find("Journal")
    if journal_el is not None:
        journal = _text(journal_el.find("Title")) or _text(journal_el.find("ISOAbbreviation"))
        pub_date = journal_el.find("JournalIssue/PubDate")
        if pub_date is not None:
            year = _text(pub_date.find("Year"))
            if not year:
                medline_date = _text(pub_date.find("MedlineDate"))
                year = medline_date[:4] if medline_date else ""

    publication_types = [
        _text(pt_el) for pt_el in article.findall("PublicationTypeList/PublicationType") if _text(pt_el)
    ]

    population = classify_population(f"{title}\n{abstract}")

    return {
        "pmid": pmid,
        "title": title,
        "journal": journal,
        "year": year,
        "publication_type": "; ".join(publication_types),
        "population": population,
        "abstract": abstract,
    }


def fetch_pubmed_abstracts(pmids: List[str]) -> List[Dict]:
    """
    Run efetch in batches and parse the returned XML into document dicts.
    """
    documents: List[Dict] = []
    for i in range(0, len(pmids), EFETCH_BATCH_SIZE):
        batch = pmids[i : i + EFETCH_BATCH_SIZE]
        params = _request_params({
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
        })
        resp = requests.post(f"{EUTILS_BASE}/efetch.fcgi", data=params, timeout=60)
        resp.raise_for_status()
        _throttle()

        root = ET.fromstring(resp.content)
        for article_el in root.findall("PubmedArticle"):
            doc = _parse_article(article_el)
            if doc:
                documents.append(doc)

    return documents


def _write_json(documents: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)


def _write_raw_csv(documents: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for doc in documents:
            writer.writerow({field: doc.get(field, "") for field in RAW_FIELDS})


def pull_and_cache_pubmed(
    fetch_limit: int = FETCH_LIMIT,
    out_json: str = RAW_JSON_PATH,
    out_csv: str = RAW_CSV_PATH,
    out_meta: str = FETCH_META_PATH,
) -> List[Dict]:
    """
    Search + fetch PubMed abstracts across all configured topics (deduped by
    PMID, capped at fetch_limit total), and write the raw results to JSON and
    CSV before anything is embedded or sent to Pinecone. Also records the
    exact fetch date and query parameters used (out_meta), since PubMed's
    result set for the same query shifts over time as new articles are
    indexed — this is the record needed to explain/reproduce a given corpus.
    """
    fetched_at = datetime.now(timezone.utc).isoformat()
    mindate, maxdate = _date_range_strings()
    per_topic_max = max(fetch_limit // len(TOPICS), 1)

    queries_run = {}
    seen_pmids = set()
    all_documents: List[Dict] = []

    for topic_key, topic_fragment in TOPICS.items():
        remaining = fetch_limit - len(all_documents)
        if remaining <= 0:
            break

        query = build_topic_query(topic_fragment)
        queries_run[topic_key] = query
        retmax = min(per_topic_max, remaining)
        pmids = search_pubmed(query, retmax=retmax, mindate=mindate, maxdate=maxdate)
        new_pmids = [p for p in pmids if p not in seen_pmids][:remaining]
        seen_pmids.update(new_pmids)

        if not new_pmids:
            continue

        docs = fetch_pubmed_abstracts(new_pmids)
        for doc in docs:
            doc["topic"] = topic_key
        all_documents.extend(docs)

    all_documents = all_documents[:fetch_limit]

    _write_json(all_documents, out_json)
    _write_raw_csv(all_documents, out_csv)

    fetch_meta = {
        "fetched_at": fetched_at,
        "fetch_limit_requested": fetch_limit,
        "abstracts_fetched": len(all_documents),
        "date_range": {"mindate": mindate, "maxdate": maxdate, "date_range_years": DATE_RANGE_YEARS},
        "publication_types": PUBLICATION_TYPES,
        "diabetes_scope": DIABETES_SCOPE,
        "topic_queries": queries_run,
        "pmids": [doc["pmid"] for doc in all_documents],
    }
    os.makedirs(os.path.dirname(out_meta) or ".", exist_ok=True)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(fetch_meta, f, ensure_ascii=False, indent=2)

    return all_documents


def load_cached_pubmed(path: str = RAW_JSON_PATH) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
