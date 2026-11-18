"""Evaluation on a mention table (one row per mention; built by scripts/02_build_tables.py and
scripts/03_run_llm.py).

Required columns:
  doc_id, matched (bool: span+type match a gold mention), gold_ids (list), ret_id, llm_id, llm_valid,
  llm_prompt_tokens, llm_completion_tokens, llm_latency_s
plus a per-document gold count table: doc_id, n_gold.

Gold mentions (the paper): accuracy@1, the share of gold mentions whose predicted identifier is one of the
gold identifiers; NIL counts as wrong (metric "acc1").
Predicted mentions (optional end-to-end setting): strict micro-F1, where a prediction is a true positive when
its span and type match a gold mention and its identifier is one of the gold identifiers; NIL predictions are
dropped (metric "f1").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NIL = "NIL"


def outcome_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    gold = df["gold_ids"].apply(set)
    df["matched"] = df["matched"].astype(bool)
    df["ret_ok"] = np.array([m and r in g for m, r, g in zip(df["matched"], df["ret_id"], gold)], dtype=bool)
    llm_final = np.where(df["llm_valid"].astype(bool), df["llm_id"], df["ret_id"])   # invalid answer -> fall back
    df["llm_final"] = llm_final
    df["llm_ok"] = np.array([(m and l in g) or (not m and l == NIL)
                             for m, l, g in zip(df["matched"], llm_final, gold)], dtype=bool)
    # retriever never outputs NIL, so a spurious mention is always an error for the retriever
    return df


def final_ids(df: pd.DataFrame, route: np.ndarray) -> np.ndarray:
    return np.where(route, df["llm_final"], df["ret_id"])


def prf(df: pd.DataFrame, route: np.ndarray, n_gold: int) -> tuple[float, float, float]:
    pred = final_ids(df, route)
    emitted = pred != NIL
    gold = df["gold_ids"].apply(set)
    tp = sum(1 for e, m, p, g in zip(emitted, df["matched"], pred, gold) if e and m and p in g)
    n_pred = int(emitted.sum())
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return 100 * p, 100 * r, 100 * f


def cost(df: pd.DataFrame, route: np.ndarray, active_params_b: float | None = None) -> dict:
    """Cost of a routing per 1,000 mentions. Calls and tokens are hardware-independent; estimated compute
    (~2 x active parameters x tokens, in TFLOPs) makes models of different sizes comparable; latency and
    accelerator-seconds depend on the machine."""
    sub = df[route]
    n = max(len(df), 1)
    tokens = sub["llm_prompt_tokens"].fillna(0).sum() + sub["llm_completion_tokens"].fillna(0).sum()
    return {
        "tflops_per_1k": (2 * active_params_b * 1e9 * tokens / 1e12) * 1000 / n if active_params_b else float("nan"),
        "share_llm": 100 * route.mean() if len(route) else 0.0,
        "tokens_per_1k": 1000 * (sub["llm_prompt_tokens"].fillna(0).sum() + sub["llm_completion_tokens"].fillna(0).sum()) / n,
        "latency_s_per_1k": 1000 * sub["llm_latency_s"].fillna(0).sum() / n,
        "gpu_s_per_1k": 1000 * sub["llm_gpu_s"].fillna(0).sum() / n if "llm_gpu_s" in sub else float("nan"),
    }


def skips(df: pd.DataFrame, route: np.ndarray) -> dict:
    skipped = ~route
    return {
        "harmful_skips": int((skipped & df["llm_ok"] & ~df["ret_ok"]).sum()),
        "beneficial_skips": int((skipped & df["ret_ok"] & ~df["llm_ok"]).sum()),
    }


def accuracy_at_1(df: pd.DataFrame, route: np.ndarray) -> tuple[float, float]:
    """Normalization accuracy over mentions that match a gold span (NIL counts as wrong), and the share of
    those mentions answered NIL. The right metric for the gold-span setting."""
    pred = final_ids(df, route)
    m = df["matched"].to_numpy()
    if not m.any():
        return float("nan"), float("nan")
    gold = df["gold_ids"].apply(set).to_numpy()
    ok = np.array([p in g for p, g in zip(pred, gold)])
    return 100 * ok[m].mean(), 100 * (pred[m] == NIL).mean()


def score(df: pd.DataFrame, route: np.ndarray, n_gold: int, metric: str = "f1") -> float:
    """The headline metric: strict end-to-end F1 ("f1") or normalization accuracy@1 ("acc1", gold spans)."""
    return accuracy_at_1(df, route)[0] if metric == "acc1" else prf(df, route, n_gold)[2]


def report(df: pd.DataFrame, route: np.ndarray, n_gold: int, name: str, active_params_b: float | None = None) -> dict:
    p, r, f = prf(df, route, n_gold)
    acc, nil = accuracy_at_1(df, route)
    return {"system": name, "P": p, "R": r, "F1": f, "acc1": acc, "nil_share": nil,
            **cost(df, route, active_params_b), **skips(df, route)}


def sweep(df: pd.DataFrame, scores: np.ndarray, n_gold: int, steps: int = 51, metric: str = "f1") -> pd.DataFrame:
    """F1 as the share of mentions sent to the LLM goes from 0 to 100% (highest gate scores first)."""
    order = np.argsort(-scores, kind="stable")
    rows = []
    for share in np.linspace(0, 1, steps):
        k = int(round(share * len(df)))
        route = np.zeros(len(df), bool)
        route[order[:k]] = True
        rows.append({"share_llm": 100 * share, "F1": score(df, route, n_gold, metric)})   # column = headline metric
    return pd.DataFrame(rows)


def auc_fc(curve: pd.DataFrame) -> float:
    """Area under F1 vs LLM share, normalised to [0, 100]."""
    x, y = curve["share_llm"].to_numpy() / 100, curve["F1"].to_numpy()
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


def _per_doc(df: pd.DataFrame, route: np.ndarray, doc_idx: np.ndarray, n_docs: int) -> dict:
    """Additive per-document counts behind both metrics, so a bootstrap sample is a weighted sum."""
    pred = final_ids(df, route)
    gold = df["gold_ids"].apply(set).to_numpy()
    matched = df["matched"].to_numpy(bool)
    ok = np.array([p in g for p, g in zip(pred, gold)]) & matched
    emitted = pred != NIL
    keep = doc_idx >= 0
    agg = lambda x: np.bincount(doc_idx[keep], weights=x[keep].astype(float), minlength=n_docs)
    return {"ok": agg(ok), "matched": agg(matched), "tp": agg(ok & emitted), "emitted": agg(emitted)}


def _metric_from_counts(c: dict, gold: np.ndarray, metric: str) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        if metric == "acc1":
            return 100 * c["ok"] / c["matched"]
        p = np.where(c["emitted"] > 0, c["tp"] / c["emitted"], 0.0)
        r = np.where(gold > 0, c["tp"] / gold, 0.0)
        return 100 * np.where(p + r > 0, 2 * p * r / (p + r), 0.0)


def paired_bootstrap(df: pd.DataFrame, route_a, route_b, gold_counts: pd.Series, n: int = 10000, seed: int = 0,
                     metric: str = "f1"):
    """Two-sided paired bootstrap over documents for metric(A) - metric(B). Returns (delta, p_value).
    Vectorised: every metric is a ratio of per-document counts, so each resample is a weighted sum of
    those counts (weights = how often each document was drawn). Same statistic as resampling the rows."""
    rng = np.random.default_rng(seed)
    docs = gold_counts.index.astype(str).to_numpy()
    pos = {d: i for i, d in enumerate(docs)}
    doc_idx = df["doc_id"].astype(str).map(pos).fillna(-1).astype(int).to_numpy()
    D = len(docs)
    base = score(df, route_a, gold_counts.sum(), metric) - score(df, route_b, gold_counts.sum(), metric)
    ca, cb = _per_doc(df, route_a, doc_idx, D), _per_doc(df, route_b, doc_idx, D)
    g = gold_counts.to_numpy(float)
    diffs = []
    for start in range(0, n, 2000):                       # chunks keep memory small for large corpora
        w = rng.multinomial(D, np.full(D, 1 / D), size=min(2000, n - start)).astype(float)
        sa = {k: w @ v for k, v in ca.items()}
        sb = {k: w @ v for k, v in cb.items()}
        gw = w @ g
        diffs.append(_metric_from_counts(sa, gw, metric) - _metric_from_counts(sb, gw, metric))
    diffs = np.concatenate(diffs)
    diffs = diffs[~np.isnan(diffs)]
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return base, min(p, 1.0)


def ece(prob: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (prob >= a) & ((prob < b) if b < 1 else (prob <= b))
        if m.any():
            e += m.mean() * abs(prob[m].mean() - y[m].mean())
    return float(e)
