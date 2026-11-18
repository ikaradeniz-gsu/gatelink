"""Glue code: build the per-mention table for one corpus split.

table columns: uid, doc_id, start, end, text, etype, matched, gold_ids, query, candidates (list of
[cid, name, score]), ret_id, <features>, and later llm_* columns added by scripts/03_run_llm.py.
"""
from __future__ import annotations

import pandas as pd

from .features import extract
from .schema import NIL, Document, Mention


def canonical_gold(docs: list[Document], kbs: dict) -> None:
    """Map gold ids that the KB lists as alternative ids (e.g. OMIM ids merged into a MeSH concept in
    CTD/MEDIC) to the KB's concept id, keeping the original ids as well."""
    for d in docs:
        for m in d.mentions:
            kb = kbs.get(m.etype)
            if kb is not None and kb.alt_to_primary:
                extra = [kb.alt_to_primary[g] for g in m.gold_ids if g in kb.alt_to_primary]
                m.gold_ids = list(dict.fromkeys(m.gold_ids + extra))


def match_gold(pred: list[Mention], docs: list[Document]) -> None:
    gold = {(m.doc_id, m.start, m.end, m.etype): m.gold_ids for d in docs for m in d.mentions}
    for m in pred:
        m.gold_ids = gold.get((m.doc_id, m.start, m.end, m.etype), [])


def build_table(docs: list[Document], pred: list[Mention], retrievers: dict, k: int, kbs: dict,
                tie_keys: dict | None = None) -> pd.DataFrame:
    """retrievers / kbs: etype -> DenseRetriever / KB. tie_keys: etype -> tie-breaking key (species step for genes)."""
    from .retrieve import generate_candidates

    canonical_gold(docs, kbs)
    match_gold(pred, docs)
    texts = {d.doc_id: d.text for d in docs}
    rows = []
    for etype, retr in retrievers.items():
        ms = [m for m in pred if m.etype == etype]
        if not ms:
            continue
        cands, queries = generate_candidates(ms, texts, retr, k=k, tie_key=(tie_keys or {}).get(etype))
        feats = extract(ms, cands, queries, kb_for=lambda t: kbs[t])
        for m, cs, q, f in zip(ms, cands, queries, feats):
            rows.append({**m.to_dict(), "matched": bool(m.gold_ids), "query": q,
                         "candidates": [[c.concept_id, c.name, c.score] for c in cs],
                         "ret_id": cs[0].concept_id if cs else NIL, **f})
    return pd.DataFrame(rows)


def gold_counts(docs: list[Document]) -> pd.DataFrame:
    """Gold mentions per document and entity type (doc_id index, one column per type)."""
    rows = {d.doc_id: pd.Series([m.etype for m in d.mentions], dtype=str).value_counts() for d in docs}
    return pd.DataFrame(rows).T.fillna(0).astype(int)
