"""Signal extractor phi (Table 1 of the manuscript). Pure Python/numpy: no model calls."""
from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np

from .kb import KB, norm_name
from .schema import Candidate, Mention

GREEK = re.compile(r"[α-ωΑ-Ω]|alpha|beta|gamma|delta|kappa", re.I)

FEATURE_GROUPS = {
    "retrieval": ["top1", "margin", "n_within_delta", "score_entropy"],
    "lexical": ["exact_any", "exact_top1", "len_chars", "len_tokens", "is_abbrev", "has_digit", "has_greek"],
    "ontology": ["top12_related", "homonymy"],
    "recognition": ["span_prob", "boundary_entropy", "type_prob"],
    "context": ["doc_consistent"],      # entity type is added as one-hot columns (etype_*)
}
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


def _entropy(scores: list[float], temp: float = 0.05) -> float:
    if len(scores) < 2:
        return 0.0
    z = np.array(scores) / temp
    p = np.exp(z - z.max())
    p /= p.sum()
    return float(-(p * np.log(p + 1e-12)).sum() / math.log(len(p)))


def extract(mentions: list[Mention], cands: list[list[Candidate]], queries: list[str], kb_for,
            delta: float = 0.02) -> list[dict]:
    """kb_for(etype) -> KB. Returns one feature dict per mention."""
    rows = []
    # document consistency: same normalised surface form linked confidently elsewhere in the doc
    confident = defaultdict(dict)
    for m, cs in zip(mentions, cands):
        if cs and (len(cs) == 1 or cs[0].score - cs[1].score > 0.1) and cs[0].score > 0.9:
            confident[m.doc_id][norm_name(m.text)] = cs[0].concept_id

    for m, cs, q in zip(mentions, cands, queries):
        kb: KB = kb_for(m.etype)
        exact = kb.exact_index()
        s = [c.score for c in cs]
        top1 = s[0] if s else 0.0
        top2 = s[1] if len(s) > 1 else 0.0
        nq = norm_name(q)
        ex_ids = exact.get(nq, set())
        r = {
            "top1": top1,
            "margin": top1 - top2,
            "n_within_delta": sum(1 for x in s if top1 - x <= delta),
            "score_entropy": _entropy(s),
            "exact_any": float(bool(ex_ids)),
            "exact_top1": float(bool(cs) and cs[0].concept_id in ex_ids),
            "len_chars": len(m.text),
            "len_tokens": len(m.text.split()),
            "is_abbrev": float(m.text.isupper() and len(m.text) <= 6 or q != m.text),
            "has_digit": float(any(ch.isdigit() for ch in m.text)),
            "has_greek": float(bool(GREEK.search(m.text))),
            "top12_related": float(len(cs) > 1 and kb.related(cs[0].concept_id, cs[1].concept_id)),
            "homonymy": len(ex_ids),
            "span_prob": m.span_prob,
            "boundary_entropy": m.boundary_entropy,
            "type_prob": m.type_prob,
            "doc_consistent": float(bool(cs) and confident[m.doc_id].get(norm_name(m.text)) == cs[0].concept_id),
            "etype": m.etype,
        }
        rows.append(r)
    return rows
