"""Train the gate on dev, apply every compared system to test, and write all result tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import evaluate as ev
from . import gate as G
from .features import ALL_FEATURES, FEATURE_GROUPS


def load(table_pkl: str | Path, only_etype: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    p = Path(table_pkl)
    df = pd.read_pickle(p)
    by_type = pd.read_csv(p.with_name(p.stem + "_gold.csv"), index_col=0)
    by_type.index = by_type.index.astype(str)
    if only_etype:
        df = df[df["etype"] == only_etype]
        by_type = by_type[[only_etype]] if only_etype in by_type else by_type.iloc[:, :0]
    df = ev.outcome_columns(df.reset_index(drop=True))
    df["doc_id"] = df["doc_id"].astype(str)
    df.attrs["gold_by_type"] = {t: int(by_type[t].sum()) for t in by_type.columns}
    return df, by_type.sum(axis=1)


def run(dev_pkl, test_pkl, gcfg: dict, out_dir, seed: int = 0, etypes: list[str] | None = None,
        bootstrap: int = 10000, only_etype: str | None = None, metric: str = "f1",
        active_params_b: float | None = None, n_random: int = 20,
        budgets: tuple[float, ...] = (0.025, 0.05, 0.10, 0.20, 0.30)) -> dict:
    """metric: "acc1" for gold mentions (accuracy@1, the paper), "f1" for predicted mentions."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev, gdev = load(dev_pkl, only_etype)
    test, gtest = load(test_pkl, only_etype)
    etypes = etypes or sorted(set(dev["etype"]) | set(test["etype"]))
    ng_dev, ng_test = int(gdev.sum()), int(gtest.sum())
    X_dev, X_test = G.design_matrix(dev, etypes=etypes), G.design_matrix(test, etypes=etypes)

    model, oof = G.fit_gate(dev, X_dev, gcfg["kind"], gcfg["target"], seed)
    s_test = model.predict_proba(X_test)[:, 1]

    taus = {
        "S6 GateLink-A": G.tau_accuracy(dev, oof, ng_dev, gcfg["accuracy_delta"], metric),
        "S7 GateLink-B": G.tau_budget(oof, gcfg["budget"]),
        "S8 GateLink-R": G.tau_conformal(dev, oof, gcfg["epsilon"]),
    }
    # baseline gates use the same accuracy-target rule on their own scores
    m_dev, m_test = G.margin_scores(dev), G.margin_scores(test)
    q_dev, q_test = G.qpp_scores(dev), G.qpp_scores(test)
    tau_m = G.tau_accuracy(dev, m_dev, ng_dev, gcfg["accuracy_delta"], metric)
    tau_q = G.tau_accuracy(dev, q_dev, ng_dev, gcfg["accuracy_delta"], metric)

    n = len(test)
    routes = {
        "S1 Retriever only": np.zeros(n, bool),
        "S2 Always-on LLM": np.ones(n, bool),
        "S3 Margin gate": m_test >= tau_m,
        "S4 QPP-style gate": q_test >= tau_q,
    }
    for name, tau in taus.items():
        routes[name] = s_test >= tau
    routes["S5 Random gate"] = G.random_route(n, routes["S6 GateLink-A"].mean(), seed)
    # S5 is one draw; the mean and sd over n_random draws at the same share go to summary.json
    rnd = [ev.score(test, G.random_route(n, routes["S6 GateLink-A"].mean(), 1000 + i), ng_test, metric)
           for i in range(n_random)]
    routes["S9 Oracle gate"] = G.oracle_route(test)
    routes = dict(sorted(routes.items()))

    # ---- Tables 3 and 4: accuracy, share of LLM calls, significance against S1 and S2
    rows = []
    for name, r in routes.items():
        rep = ev.report(test, r, ng_test, name, active_params_b)
        if bootstrap and name != "S2 Always-on LLM":
            rep["delta_vs_S2"], rep["p_vs_S2"] = ev.paired_bootstrap(test, r, routes["S2 Always-on LLM"], gtest,
                                                                     n=bootstrap, seed=seed, metric=metric)
        if bootstrap and name != "S1 Retriever only":
            rep["delta_vs_S1"], rep["p_vs_S1"] = ev.paired_bootstrap(test, r, routes["S1 Retriever only"], gtest,
                                                                     n=bootstrap, seed=seed, metric=metric)
        rows.append(rep)
    main = pd.DataFrame(rows)
    main["calls_saved_vs_S2"] = 100 - main["share_llm"]
    main.to_csv(out / "main.csv", index=False)

    # ---- Figure 2: accuracy against the share of mentions sent to the LLM
    curves = []
    for name, sc in {"GateLink": s_test, "Margin gate": m_test, "QPP-style gate": q_test,
                     "Oracle gate": G.oracle_route(test).astype(float)}.items():
        c = ev.sweep(test, sc, ng_test, metric=metric)
        c["system"] = name
        curves.append(c)
    # random gate: mean over n_random draws (its spread is reported in random_draws.csv)
    rand = [ev.sweep(test, np.random.default_rng(1000 + i).random(n), ng_test, metric=metric).assign(draw=i)
            for i in range(n_random)]
    rand = pd.concat(rand)
    rand.to_csv(out / "random_draws.csv", index=False)
    rmean = rand.groupby("share_llm", as_index=False)["F1"].agg(["mean", "std"]).rename(columns={"mean": "F1", "std": "F1_sd"})
    curves.append(rmean.assign(system="Random gate"))
    curves = pd.concat(curves)
    curves.to_csv(out / "curves.csv", index=False)
    auc = curves.groupby("system").apply(ev.auc_fc, include_groups=False).rename("AUC_FC")
    auc.to_csv(out / "auc_fc.csv")

    # per-mention gate scores (lets any comparison be recomputed without refitting)
    pd.DataFrame({"uid": test["uid"] if "uid" in test else np.arange(n), "doc_id": test["doc_id"],
                  "gatelink": s_test, "margin": m_test, "qpp": q_test}).to_csv(out / "scores.csv", index=False)

    # ---- Supplementary Table S3: matched-budget comparison, the same share of mentions sent to the LLM, top-k by each gate's score
    mb = []
    order = {k: np.argsort(-v, kind="stable") for k, v in {"GateLink": s_test, "Margin gate": m_test,
                                                           "QPP-style gate": q_test}.items()}
    def top(name, share):
        r = np.zeros(n, bool)
        r[order[name][:int(round(share * n))]] = True
        return r
    for share in budgets:
        g = top("GateLink", share)
        for other in ("QPP-style gate", "Margin gate"):
            o = top(other, share)
            d, p = ev.paired_bootstrap(test, g, o, gtest, n=bootstrap, seed=seed, metric=metric) if bootstrap \
                else (ev.score(test, g, ng_test, metric) - ev.score(test, o, ng_test, metric), np.nan)
            mb.append({"share_llm": 100 * share, "comparison": f"GateLink vs {other}",
                       "score_GateLink": ev.score(test, g, ng_test, metric),
                       "score_other": ev.score(test, o, ng_test, metric), "delta": d, "p": p,
                       "routed_by_both": int((g & o).sum()), "routed_by_one_only": int((g ^ o).sum())})
    pd.DataFrame(mb).to_csv(out / "matched_budget.csv", index=False)

    # ---- Section 3.4: results per entity type (GateLink-A)
    per_type = []
    r_a = routes["S6 GateLink-A"]
    for t in etypes:
        mask = (test["etype"] == t).to_numpy()
        if not mask.any():
            continue
        sub = test[mask].reset_index(drop=True)
        ng_t = test.attrs["gold_by_type"].get(t, 0)
        r_r = routes["S8 GateLink-R"]
        per_type.append({"etype": t, "metric": metric, "n": len(sub),
                         "S1": ev.score(sub, np.zeros(len(sub), bool), ng_t, metric),
                         "S2": ev.score(sub, np.ones(len(sub), bool), ng_t, metric),
                         "GateLinkA": ev.score(sub, r_a[mask], ng_t, metric),
                         "GateLinkR": ev.score(sub, r_r[mask], ng_t, metric),
                         "share_llm_R": 100 * r_r[mask].mean(),
                         "share_llm": 100 * r_a[mask].mean(),
                         "share_llm_oracle": 100 * routes["S9 Oracle gate"][mask].mean(),
                         **ev.skips(sub, r_a[mask])})
    pd.DataFrame(per_type).to_csv(out / "per_type.csv", index=False)

    # ---- Supplementary Table S4: feature-group ablation (out-of-fold development scores)
    y_dev = G.labels(dev, gcfg["target"])
    abl = []
    variants = {"All": ALL_FEATURES, "Retrieval only": FEATURE_GROUPS["retrieval"]}
    for g, fs in FEATURE_GROUPS.items():
        variants[f"- {g}"] = [f for f in ALL_FEATURES if f not in fs]
    for name, fs in variants.items():
        Xd = G.design_matrix(dev, fs, etypes if name != "- context" else None)
        _, o = G.fit_gate(dev, Xd, gcfg["kind"], gcfg["target"], seed)
        tau = G.tau_accuracy(dev, o, ng_dev, gcfg["accuracy_delta"], metric)
        abl.append({"features": name, "AUROC": roc_auc_score(y_dev, o) if len(set(y_dev)) > 1 else np.nan,
                    "ECE": ev.ece(o, y_dev),
                    "score_at_30pct": ev.sweep(dev, o, ng_dev, metric=metric).set_index("share_llm").loc[30.0, "F1"],
                    "share_llm_at_delta": 100 * (o >= tau).mean()})
    pd.DataFrame(abl).to_csv(out / "ablation.csv", index=False)

    # ---- Supplementary Table S5: risk control, target against realized share of harmful skips
    crc = []
    always_ok = test["llm_ok"].to_numpy()
    for eps in (0.0025, 0.005, 0.01, 0.02):
        tau = G.tau_conformal(dev, oof, eps)
        r = s_test >= tau
        realised = ((~r) & always_ok & ~test["ret_ok"].to_numpy()).mean()
        crc.append({"epsilon": eps, "realised_loss": realised, "share_llm": 100 * r.mean(),
                    "score": ev.score(test, r, ng_test, metric), "met": realised <= eps})
    pd.DataFrame(crc).to_csv(out / "conformal.csv", index=False)

    # ---- Supplementary Section S6: calibration of the gate
    y_test = G.labels(test, gcfg["target"])
    pd.DataFrame({"prob": s_test, "y": y_test, "etype": test["etype"]}).to_csv(out / "calibration.csv", index=False)

    summary = {"metric": metric, "taus": {**taus, "S3": tau_m, "S4": tau_q}, "n_test_mentions": n, "n_gold_test": ng_test,
               "gate_auroc_test": roc_auc_score(y_test, s_test) if len(set(y_test)) > 1 else None,
               "gate_ece_test": ev.ece(s_test, y_test),
               "random_gate_at_S6_share": {"mean": float(np.mean(rnd)), "sd": float(np.std(rnd, ddof=1)),
                                           "n_draws": n_random}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary
