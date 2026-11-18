"""Build the tables of the paper and the Supplement (markdown) from the result folders of 04_evaluate.py.

python scripts/05_paper_tables.py --results runs/gatelink/results --out runs/gatelink/paper_tables.md
python scripts/05_paper_tables.py --results results --out paper_tables.md      # the released results

Main-text values are means over the three cross-validation splits (seeds 0-2). A value is marked as not
significantly different from the retriever (†) unless p < 0.05 in all three splits.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

CORPORA = [("ncbi_disease", "NCBI Disease"), ("bc5cdr", "BC5CDR"), ("nlm_chem", "NLM-Chem"), ("nlm_gene_sp", "NLM-Gene")]
LLMS = [("small", "gemma4:12b"), ("mid", "gemma4:26b"), ("large", "gemma4:31b")]
SYSTEMS = ["S1 Retriever only", "S2 Always-on LLM", "S3 Margin gate", "S4 QPP-style gate", "S5 Random gate",
           "S6 GateLink-A", "S7 GateLink-B", "S8 GateLink-R", "S9 Oracle gate"]
SPLITS = range(3)


def run_dir(root: Path, corpus: str, llm: str, split: int = 0) -> Path | None:
    d = root / f"{corpus}_gold_{llm}_s{split}"
    return d if d.exists() else None


def load_main(root: Path) -> pd.DataFrame:
    rows = []
    for d in root.glob("*_gold_*_s*"):
        m = re.match(r"(.+)_gold_(\w+)_s(\d)$", d.name)
        if not m or "-to-" in m.group(1):
            continue
        df = pd.read_csv(d / "main.csv")
        df["corpus"], df["llm"], df["seed"] = m.group(1), m.group(2), int(m.group(3))
        s = json.loads((d / "summary.json").read_text())
        df["auroc"], df["ece"] = s["gate_auroc_test"], s["gate_ece_test"]
        rows.append(df)
    return pd.concat(rows)


def aggregate(all_main: pd.DataFrame, corpus: str, llm: str) -> pd.DataFrame | None:
    g = all_main[(all_main.corpus == corpus) & (all_main.llm == llm)]
    if g.empty:
        return None
    return g.groupby("system").agg(acc=("acc1", "mean"), sd=("acc1", "std"), share=("share_llm", "mean"),
                                   p1=("p_vs_S1", "max"), p2=("p_vs_S2", "max"), d1=("delta_vs_S1", "mean"),
                                   d2=("delta_vs_S2", "mean"), n=("seed", "count"))


def cell(t, system, share=True, dagger=False) -> str:
    """accuracy@1 [± SD over splits] [† not significant against S1] [(share of mentions sent to the LLM)]."""
    if t is None or system not in t.index:
        return "-"
    r = t.loc[system]
    x = f"{r.acc:.2f}"
    if r.n > 1 and r.sd > 0.005:
        x += f" ± {r.sd:.2f}"
    if dagger and system != "S1 Retriever only" and (pd.isna(r.p1) or r.p1 >= 0.05):
        x += "†"
    if share and system not in ("S1 Retriever only", "S2 Always-on LLM"):
        x += f" ({r.share:.1f})"
    return x


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs/gatelink/results")
    ap.add_argument("--out", default="runs/gatelink/paper_tables.md")
    a = ap.parse_args()
    root = Path(a.results)
    A = load_main(root)
    sec = []

    # ---- Table 3: all systems with gemma4:26b
    T = {c: aggregate(A, c, "mid") for c, _ in CORPORA}
    sec.append(("Table 3: accuracy@1 (%) with gemma4:26b (share of mentions sent to the LLM)",
                table(["System"] + [n for _, n in CORPORA],
                      [[s] + [cell(T[c], s, dagger=True) for c, _ in CORPORA] for s in SYSTEMS])))

    # ---- Table 4: effect of reranker size
    rows = []
    for c, cn in CORPORA:
        for l, ln in LLMS:
            t = aggregate(A, c, l)
            if t is not None:
                rows.append([f"{cn} ({t.loc['S1 Retriever only'].acc:.2f})", ln, cell(t, "S2 Always-on LLM", False),
                             cell(t, "S6 GateLink-A"), cell(t, "S8 GateLink-R"), cell(t, "S9 Oracle gate")])
    sec.append(("Table 4: effect of reranker size", table(
        ["Corpus (S1)", "Reranker", "S2 Always-on", "S6 GateLink-A", "S8 GateLink-R", "S9 Oracle"], rows)))

    # ---- significance: largest p over the three splits
    rows = []
    for c, cn in CORPORA:
        for l, ln in LLMS:
            t = aggregate(A, c, l)
            if t is None:
                continue
            s2, ga, gr = t.loc["S2 Always-on LLM"], t.loc["S6 GateLink-A"], t.loc["S8 GateLink-R"]
            rows.append([cn, ln, f"{s2.d1:+.2f} ({s2.p1:.3f})", f"{ga.d1:+.2f} ({ga.p1:.3f})",
                         f"{ga.d2:+.2f} ({ga.p2:.3f})", f"{gr.d1:+.2f} ({gr.p1:.3f})", f"{gr.d2:+.2f} ({gr.p2:.3f})"])
    sec.append(("Differences in accuracy@1 points (largest p over the three splits)", table(
        ["Corpus", "Reranker", "S2 − S1", "GateLink-A − S1", "GateLink-A − S2", "GateLink-R − S1", "GateLink-R − S2"],
        rows)))

    # ---- gate discrimination and calibration on test (split 1)
    rows = [[c, l, f"{g.auroc.iloc[0]:.3f}", f"{g.ece.iloc[0]:.3f}"]
            for (c, l), g in A[A.seed == 0].groupby(["corpus", "llm"])]
    sec.append(("Gate AUROC and expected calibration error on test (first split)",
                table(["Corpus", "Reranker", "AUROC", "ECE"], rows)))

    # ---- Supplementary Table S3: matched budgets and area under the accuracy-share curve
    rows, rows_auc = [], []
    for c, cn in CORPORA:
        for l, ln in LLMS:
            d = run_dir(root, c, l)
            if not d:
                continue
            mb = pd.read_csv(d / "matched_budget.csv")
            for other, on in [("QPP-style gate", "QPP-style"), ("Margin gate", "Margin")]:
                m = mb[mb.comparison == f"GateLink vs {other}"].sort_values("share_llm")
                rows.append([cn, ln, on] + [f"{r.delta:+.2f}{'*' if r.p < 0.05 else ''}" for r in m.itertuples()])
            auc = pd.read_csv(d / "auc_fc.csv").set_index("system").AUC_FC
            rows_auc.append([cn, ln] + [f"{auc[s]:.2f}" for s in
                                        ("GateLink", "QPP-style gate", "Margin gate", "Random gate", "Oracle gate")])
    sec.append(("Supplementary Table S3a: GateLink minus a training-free gate at matched budgets (* p < 0.05)",
                table(["Corpus", "Reranker", "vs", "2.5%", "5%", "10%", "20%", "30%"], rows)))
    sec.append(("Supplementary Table S3b: area under the accuracy-share curve",
                table(["Corpus", "Reranker", "GateLink", "QPP-style", "Margin", "Random", "Oracle"], rows_auc)))

    # ---- Supplementary Table S4: feature-group ablation (AUROC / ECE, gemma4:26b runs, first split)
    abl = {c: pd.read_csv(run_dir(root, c, "mid") / "ablation.csv").set_index("features") for c, _ in CORPORA}
    names = {"All": "All features", "Retrieval only": "Retrieval only", "- retrieval": "All − retrieval",
             "- lexical": "All − lexical", "- ontology": "All − ontology", "- context": "All − context"}
    rows = [[n] + [f"{abl[c].loc[k, 'AUROC']:.3f} / {abl[c].loc[k, 'ECE']:.3f}" for c, _ in CORPORA]
            for k, n in names.items()]
    sec.append(("Supplementary Table S4: feature-group ablation, AUROC / ECE",
                table(["Features"] + [n for _, n in CORPORA], rows)))

    # ---- Supplementary Table S5: risk control, target against realized share of harmful skips
    rows = []
    for c, cn in CORPORA:
        for l, ln in LLMS:
            d = run_dir(root, c, l)
            if d:
                cf = pd.read_csv(d / "conformal.csv")
                rows.append([cn, ln] + [f"{100 * r.realised_loss:.2f}{'' if r.met else '†'} ({r.share_llm:.1f}; "
                                        f"{r.score:.2f})" for r in cf.itertuples()])
    sec.append(("Supplementary Table S5: realized share of harmful skips (%) († target missed; "
                "share sent; accuracy@1)", table(["Corpus", "Reranker", "ε = 0.25%", "ε = 0.5%", "ε = 1%", "ε = 2%"],
                                                 rows)))

    # ---- Supplementary Table S7: transfer between corpora
    labels = {("ncbi_disease", "bc5cdr"): "NCBI Disease → BC5CDR (diseases)",
              ("bc5cdr", "nlm_chem"): "BC5CDR (chemicals) → NLM-Chem"}
    keys = ["S1 Retriever only", "S2 Always-on LLM", "S3 Margin gate", "S4 QPP-style gate", "S6 GateLink-A",
            "S8 GateLink-R", "S9 Oracle gate"]
    rows = []
    for d in sorted(root.glob("*-to-*_gold_*_s0")):
        src, tgt, l = re.match(r"(.+)-to-(.+)_gold_(\w+)_s0", d.name).groups()
        mm = pd.read_csv(d / "main.csv").set_index("system")

        def f(k):
            r = mm.loc[k]
            dag = "" if k.startswith("S1") or r.p_vs_S1 < 0.05 else "†"
            return f"{r.acc1:.2f}{dag}" + ("" if k.startswith(("S1", "S2")) else f" ({r.share_llm:.1f})")
        rows.append([labels.get((src, tgt), f"{src} → {tgt}"), dict(LLMS)[l]] + [f(k) for k in keys])
    sec.append(("Supplementary Table S7: transfer between corpora", table(
        ["Source → target", "Reranker", "Retriever", "Always-on", "Margin", "QPP-style", "GateLink-A", "GateLink-R",
         "Oracle"], rows)))

    # ---- Supplementary Table S9: results for each cross-validation split, with the random gate
    rows = []
    for c, cn in CORPORA:
        for l, ln in LLMS:
            for s in SPLITS:
                d = run_dir(root, c, l, s)
                if not d:
                    continue
                mm = pd.read_csv(d / "main.csv").set_index("system")
                rg = json.loads((d / "summary.json").read_text())["random_gate_at_S6_share"]
                ga, gr = mm.loc["S6 GateLink-A"], mm.loc["S8 GateLink-R"]
                rows.append([cn, ln, str(s + 1), f"{ga.acc1:.2f} ({ga.share_llm:.1f})",
                             f"{gr.acc1:.2f} ({gr.share_llm:.1f})", f"{rg['mean']:.2f} ± {rg['sd']:.2f}"])
    sec.append(("Supplementary Table S9: results for each cross-validation split", table(
        ["Corpus", "Reranker", "Split", "GateLink-A acc@1 (sent %)", "GateLink-R acc@1 (sent %)",
         "Random gate at GateLink-A share, mean ± SD over 20 draws"], rows)))

    # ---- Supplementary Table S11: precision, recall and F1 with NIL as abstention
    rows = []
    for c, cn in CORPORA:
        for l, ln in LLMS:
            ds = [d for s in SPLITS if (d := run_dir(root, c, l, s))]
            if not ds:
                continue
            m = pd.concat([pd.read_csv(d / "main.csv") for d in ds]).groupby("system")[["P", "R", "F1", "nil_share"]].mean()
            for k, n in [("S2 Always-on LLM", "Always-on"), ("S6 GateLink-A", "GateLink-A"), ("S8 GateLink-R", "GateLink-R")]:
                r = m.loc[k]
                rows.append([cn, ln, n, f"{r.P:.2f}", f"{r.R:.2f}", f"{r.F1:.2f}", f"{r.nil_share:.1f}"])
    sec.append(("Supplementary Table S11: precision, recall and F1 (NIL as abstention)", table(
        ["Corpus", "Reranker", "System", "Precision", "Recall (= acc@1)", "F1", "NIL answers (%)"], rows)))

    # ---- Supplementary Table S12: NLM-Gene without the species step (earlier run, see results/README.md)
    if run_dir(root, "nlm_gene", "small") and run_dir(root, "nlm_gene", "mid"):
        m = {l: pd.concat([pd.read_csv(run_dir(root, "nlm_gene", l, s) / "main.csv") for s in SPLITS])
             .groupby("system")[["acc1", "share_llm"]].mean() for l in ("small", "mid")}
        rows = [[k] + [f"{m[l].loc[k, 'acc1']:.2f} ({m[l].loc[k, 'share_llm']:.1f})" for l in ("small", "mid")]
                for k in SYSTEMS if k != "S5 Random gate"]
        sec.append(("Supplementary Table S12: NLM-Gene without the species step", table(
            ["System", "gemma4:12b acc@1 (sent %)", "gemma4:26b acc@1 (sent %)"], rows)))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(f"## {title}\n\n{body}" for title, body in sec) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
