"""
Patient-mode safety helpers.

- names_unmentioned_drugs: drug, drug-class and supplement names an answer uses
  that the question never mentioned. Patient mode must not introduce these; the
  eval reports every hit.
- touches_medication: whether a patient answer must end with MEDICATION_NOTE.

The term list is deliberately broad (generic names, common brands, classes,
supplements). It is a detector, not a filter: it never rewrites an answer.
"""
import re
from typing import Iterable, List

MEDICATION_NOTE = "Talk to your doctor or pharmacist before changing anything."

DRUG_TERMS = [
    # biguanides / sulfonylureas / meglitinides
    "metformin", "glucophage", "sulfonylurea", "sulfonylureas", "glipizide", "glyburide", "glibenclamide",
    "glimepiride", "gliclazide", "repaglinide", "nateglinide", "meglitinide", "meglitinides",
    # TZDs, DPP-4, SGLT2, GLP-1, dual agonists
    "thiazolidinedione", "thiazolidinediones", "pioglitazone", "rosiglitazone",
    "dpp-4", "dpp4", "dpp-4 inhibitor", "dpp-4 inhibitors", "gliptin", "gliptins", "sitagliptin", "saxagliptin",
    "linagliptin", "alogliptin", "vildagliptin",
    "sglt2", "sglt-2", "sglt2 inhibitor", "sglt2 inhibitors", "gliflozin", "gliflozins", "empagliflozin",
    "dapagliflozin", "canagliflozin", "ertugliflozin", "sotagliflozin",
    "glp-1", "glp1", "glp-1 receptor agonist", "glp-1 receptor agonists", "glp-1 ra", "glp-1 ras",
    "semaglutide", "liraglutide", "dulaglutide", "exenatide", "lixisenatide", "tirzepatide", "retatrutide",
    "ozempic", "wegovy", "rybelsus", "victoza", "saxenda", "trulicity", "byetta", "bydureon", "mounjaro", "zepbound",
    "januvia", "jardiance", "farxiga", "invokana", "actos",
    # alpha-glucosidase inhibitors, others
    "acarbose", "miglitol", "voglibose", "alpha-glucosidase inhibitor", "alpha-glucosidase inhibitors",
    "pramlintide", "colesevelam", "bromocriptine", "orlistat", "phentermine", "topiramate", "naltrexone",
    "bupropion", "statin", "statins", "atorvastatin", "rosuvastatin", "simvastatin", "warfarin",
    # insulins (named products/types; the plain word "insulin" is a hormone, not flagged)
    "insulin glargine", "insulin detemir", "insulin degludec", "insulin lispro", "insulin aspart",
    "glargine", "detemir", "degludec", "lispro", "aspart", "glulisine", "nph insulin", "lantus", "levemir",
    "tresiba", "humalog", "novolog", "basal insulin", "bolus insulin", "prandial insulin",
    # supplements
    "chromium", "cinnamon", "berberine", "magnesium supplement", "magnesium supplements", "vitamin d",
    "vitamin d supplementation", "vitamin b12", "omega-3", "fish oil", "probiotic", "probiotics", "inositol",
    "myo-inositol", "alpha-lipoic acid", "fenugreek", "curcumin", "ginseng", "zinc supplementation",
    "psyllium", "resveratrol", "coenzyme q10",
]

_GENERIC_MEDICATION_WORDS = re.compile(
    r"\b(medicines?|medications?|drugs?|insulin|pills?|tablets?|doses?|dosing|supplements?|prescri\w*)\b",
    re.IGNORECASE,
)


def _pattern(term: str) -> re.Pattern:
    return re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", re.IGNORECASE)


_PATTERNS = [(t, _pattern(t)) for t in sorted(set(DRUG_TERMS), key=len, reverse=True)]


def drug_terms_in(text: str) -> List[str]:
    """Drug/class/supplement terms present in text, longest match first, no overlaps double-counted."""
    found, remaining = [], (text or "")
    for term, pattern in _PATTERNS:
        if pattern.search(remaining):
            found.append(term)
            remaining = pattern.sub(" ", remaining)
    return found


def names_unmentioned_drugs(question: str, answer_sentences: Iterable[str]) -> List[str]:
    """Terms in the answer that the question did not mention (case-insensitive, word-bounded)."""
    asked = {t.lower() for t in drug_terms_in(question)}
    asked_text = (question or "").lower()
    out = []
    for term in drug_terms_in(" ".join(answer_sentences)):
        if term in asked or term in asked_text:
            continue
        if term not in out:
            out.append(term)
    return out


def touches_medication(question: str, answer_sentences: Iterable[str]) -> bool:
    text = f"{question} {' '.join(answer_sentences)}"
    return bool(_GENERIC_MEDICATION_WORDS.search(text) or drug_terms_in(text))
