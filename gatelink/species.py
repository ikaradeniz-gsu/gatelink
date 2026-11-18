"""Species step of the candidate generator for genes (Section 2.2).

The same gene symbol (e.g. BRCA1) names orthologous genes in many species, so exact-match and SapBERT
scores tie across species and the retriever's top-1 species is arbitrary. Like the species step of gene
normalizers such as GNormPlus, we break such ties by (1) how often the document names each species, then
(2) a fixed default order, human first. The rule uses no training data, so the retriever stays zero-shot.
Scores are never changed; only candidates with equal scores are reordered.
"""
from __future__ import annotations

import re
from collections import Counter

# taxon -> pattern of names used in PubMed text (the 12 taxa of the gene vocabulary)
TAXON_PATTERNS = {
    "9606": r"humans?|patients?|homo sapiens|people|men|women|children|adults?|infants?",
    "10090": r"mouse|mice|murine|mus musculus",
    "10116": r"rats?|rattus",
    "7227": r"drosophila|fruit fl(?:y|ies)",
    "3702": r"arabidopsis",
    "7955": r"zebrafish|danio rerio",
    "6239": r"c\. ?elegans|caenorhabditis|nematodes?",
    "559292": r"yeast|saccharomyces|cerevisiae",
    "4932": r"yeast|saccharomyces|cerevisiae",
    "9913": r"bovine|cattle|bos taurus|cows?",
    "9031": r"chickens?|chick|gallus",
    "8355": r"xenopus|frogs?",
}
_RX = {t: re.compile(rf"\b(?:{p})\b", re.I) for t, p in TAXON_PATTERNS.items()}


def doc_species(text: str) -> Counter:
    """Number of times the document names each taxon."""
    return Counter({t: len(rx.findall(text)) for t, rx in _RX.items() if rx.search(text)})


# fixed default order when the document names no species (or ties): human first, then common model organisms
DEFAULT_ORDER = ["9606", "10090", "10116", "7955", "7227", "6239", "559292", "4932", "3702", "9913", "9031", "8355"]
_RANK = {t: i for i, t in enumerate(DEFAULT_ORDER)}


def tie_breaker(texts: dict[str, str], taxon_of: dict[str, str]):
    """Returns key(mention, candidate) -> tuple; smaller sorts first among equal scores."""
    per_doc = {d: doc_species(t) for d, t in texts.items()}

    def key(m, c):
        tax = taxon_of.get(c.concept_id)
        return (-per_doc.get(m.doc_id, Counter()).get(tax, 0), _RANK.get(tax, len(_RANK)))
    return key
