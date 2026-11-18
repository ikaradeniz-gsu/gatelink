"""Gate g and the three threshold rules A, B and R (Section 2.3 of the manuscript)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import evaluate as ev
from .features import ALL_FEATURES


def design_matrix(df: pd.DataFrame, features: list[str] | None = None, etypes: list[str] | None = None) -> np.ndarray:
    feats = features or ALL_FEATURES
    X = df[feats].astype(float).to_numpy()
    if etypes:
        onehot = np.stack([(df["etype"] == t).to_numpy(float) for t in etypes], 1)
        X = np.hstack([X, onehot])
    return X


def make_model(kind: str = "hgb", seed: int = 0, cv=3):
    """kind: hgb (scikit-learn HistGradientBoosting, no system dependencies; default),
    gbm (LightGBM; falls back to hgb if it cannot be loaded), lr (logistic regression)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    hgb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                 l2_regularization=1.0, random_state=seed)
    if kind == "lr":
        base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    elif kind == "gbm":
        try:
            from lightgbm import LGBMClassifier
            base = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                  subsample=0.8, colsample_bytree=0.8, random_state=seed, verbose=-1)
        except (ImportError, OSError) as e:          # e.g. missing libomp on macOS
            import warnings
            warnings.warn(f"LightGBM unavailable ({type(e).__name__}); using HistGradientBoosting instead")
            base = hgb()
    else:
        base = hgb()
    return CalibratedClassifierCV(base, method="isotonic", cv=cv)


def labels(df: pd.DataFrame, target: str = "error") -> np.ndarray:
    """error: retriever top-1 is wrong (GateLink default). benefit: LLM fixes the retriever."""
    if target == "error":
        return (~df["ret_ok"]).to_numpy(int)
    return (df["llm_ok"] & ~df["ret_ok"]).to_numpy(int)


def _fit_calibrated(X, y, groups, kind, seed):
    """Gate with isotonic calibration; the calibration folds are grouped by document."""
    folds = list(GroupKFold(min(3, len(set(groups)))).split(X, y, groups))
    return make_model(kind, seed, cv=folds).fit(X, y)


def fit_gate(dev: pd.DataFrame, X_dev: np.ndarray, kind="hgb", target="error", seed=0):
    """Returns (fitted model, out-of-fold dev scores). OOF scores are used to choose tau.
    Folds are grouped by document: mentions of one document (often the same string repeated) never sit
    on both sides of a split, so the OOF scores are as honest as scores on unseen test documents."""
    y = labels(dev, target)
    groups = dev["doc_id"].astype(str).to_numpy()
    cv = StratifiedGroupKFold(min(5, len(set(groups))), shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in cv.split(X_dev, y, groups):
        m = _fit_calibrated(X_dev[tr], y[tr], groups[tr], kind, seed)
        oof[te] = m.predict_proba(X_dev[te])[:, 1]
    model = _fit_calibrated(X_dev, y, groups, kind, seed)
    return model, oof


# ------------------------------------------------------------------ threshold policies
def tau_budget(scores_dev: np.ndarray, budget: float) -> float:
    """Rule B: send a fixed share `budget` (0-1) of the mentions to the LLM."""
    if budget <= 0:
        return np.inf
    return float(np.quantile(scores_dev, 1 - budget))


def tau_accuracy(dev: pd.DataFrame, scores_dev: np.ndarray, n_gold: int, delta: float, metric: str = "f1") -> float:
    """Rule A: largest tau (fewest LLM calls) whose dev score is within `delta` points of the best dev score over all
    thresholds. Using the best rather than always-on keeps the rule meaningful when always-on is not the best."""
    taus = np.append(np.sort(np.unique(scores_dev))[::-1], -np.inf)    # high tau = little routing
    vals = [ev.score(dev, scores_dev >= t, n_gold, metric) for t in taus]
    target = max(vals) - delta
    for t, v in zip(taus, vals):
        if v >= target:
            return float(t)
    return float("-inf")


def tau_conformal(dev: pd.DataFrame, scores_dev: np.ndarray, epsilon: float) -> float:
    """Rule R: conformal risk control (Angelopoulos et al., ICLR 2024).
    Loss per mention at threshold tau: 1[skipped, LLM right, retriever wrong] (a harmful skip),
    which is monotone in tau. Picks the largest tau with (n/(n+1)) * mean loss + 1/(n+1) <= epsilon,
    which bounds the expected share of harmful skips by epsilon (in [0, 1])."""
    harm_if_skipped = (dev["llm_ok"] & ~dev["ret_ok"]).to_numpy()
    n = len(dev)
    best = 0.0
    for tau in np.sort(np.unique(np.append(scores_dev, np.inf))):   # ascending: loss grows with tau
        loss = (harm_if_skipped & (scores_dev < tau)).mean()
        if n / (n + 1) * loss + 1 / (n + 1) <= epsilon:
            best = float(tau)
        else:
            break
    return best


# ------------------------------------------------------------------ baseline gates (no training)
def margin_scores(df: pd.DataFrame) -> np.ndarray:
    return -df["margin"].to_numpy(float)          # small margin -> high score -> route


def qpp_scores(df: pd.DataFrame) -> np.ndarray:
    return -df["top1"].to_numpy(float)


def oracle_route(df: pd.DataFrame) -> np.ndarray:
    return (df["llm_ok"] & ~df["ret_ok"]).to_numpy()


def random_route(n: int, share: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    route = np.zeros(n, bool)
    route[rng.choice(n, size=int(round(share * n)), replace=False)] = True
    return route
