"""Figure 2 and the calibration figure of the Supplement (Section S6) from the result folders of 04_evaluate.py.

python scripts/06_figures.py --results runs/gatelink/results --out runs/gatelink/figures
python scripts/06_figures.py --results results --out figures          # the released results

Writes figure2.pdf/.png (accuracy@1 against the share of mentions sent to gemma4:26b, one panel per corpus,
178 mm wide) and figureS6.pdf/.png (reliability diagram of the gate, 86 mm wide). Colors are colorblind-safe;
gate identity is also carried by line style and labels.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CORPORA = [("ncbi_disease", "NCBI Disease"), ("bc5cdr", "BC5CDR"), ("nlm_chem", "NLM-Chem"), ("nlm_gene_sp", "NLM-Gene")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GATES = {
    "GateLink":       dict(color=BLUE, lw=2.0, ls="-", label="GateLink", z=5),
    "QPP-style gate": dict(color=ORANGE, lw=1.4, ls="--", label="QPP-style gate", z=4),
    "Margin gate":    dict(color=AQUA, lw=1.4, ls="-.", label="Margin gate", z=3),
    "Random gate":    dict(color=INK2, lw=1.0, ls=":", label="Random gate", z=2),
    "Oracle gate":    dict(color="#9a9893", lw=1.1, ls="-", label="Oracle gate (upper bound)", z=1),
}
CALIBRATION = {"ncbi_disease": (BLUE, "o"), "bc5cdr": (ORANGE, "s"), "nlm_chem": (AQUA, "^"), "nlm_gene_sp": (YELLOW, "D")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "pdf.fonttype": 42})


def figure2(root: Path, out: Path, llm: str = "mid"):
    panels = [(c, n) for c, n in CORPORA if (root / f"{c}_gold_{llm}_s0" / "curves.csv").exists()]
    fig, axes = plt.subplots(1, len(panels), figsize=(178 / 25.4, 2.6), sharex=True)
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, (c, name) in zip(axes, panels):
        d = root / f"{c}_gold_{llm}_s0"                 # first cross-validation split
        curves = pd.read_csv(d / "curves.csv")
        main = pd.read_csv(d / "main.csv").set_index("system")
        for gate, st in GATES.items():
            s = curves[curves.system == gate].sort_values("share_llm")
            ax.plot(s.share_llm, s.F1, color=st["color"], lw=st["lw"], ls=st["ls"], zorder=st["z"], label=st["label"],
                    solid_capstyle="round")         # column F1 holds the headline metric, here accuracy@1
        a = main.loc["S6 GateLink-A"]
        ax.plot([a.share_llm], [a.acc1], marker="o", ms=5, color=BLUE, mec="#fcfcfb", mew=1.0, zorder=7,
                ls="none", label="GateLink-A threshold")
        ax.set_title(name, fontsize=8, color=INK, loc="left", pad=4, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, width=0.5)
    axes[0].set_ylabel("Accuracy@1 (%)")
    fig.supxlabel("Mentions sent to the LLM (%)", fontsize=7.5, y=0.01)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=6.6, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.6)
    fig.tight_layout(rect=(0, 0.02, 1, 0.84), w_pad=1.2)
    fig.savefig(out / "figure2.pdf")
    fig.savefig(out / "figure2.png", dpi=300)
    plt.close(fig)


def figure_calibration(root: Path, out: Path, llm: str = "mid", min_bin: int = 10):
    fig, ax = plt.subplots(figsize=(86 / 25.4, 86 / 25.4))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, zorder=0)
    bins = np.linspace(0, 1, 11)
    for c, name in CORPORA:
        f = root / f"{c}_gold_{llm}_s0" / "calibration.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        b = np.clip(np.digitize(d.prob, bins) - 1, 0, 9)
        g = d.groupby(b).agg(p=("prob", "mean"), y=("y", "mean"), n=("y", "size"))
        g = g[g.n >= min_bin]                           # bins with fewer than 10 mentions are not shown
        color, marker = CALIBRATION[c]
        ax.plot(g.p, g.y, color=color, lw=1.6, marker=marker, ms=3.5, mec="#fcfcfb", mew=0.6, label=name)
    ax.set_xlabel("Predicted probability that the retriever is wrong")
    ax.set_ylabel("Observed share of retriever errors")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color=GRID, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figureS6.png", dpi=300)
    fig.savefig(out / "figureS6.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs/gatelink/results")
    ap.add_argument("--out", default="runs/gatelink/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    figure2(Path(a.results), out)
    figure_calibration(Path(a.results), out)
    print("wrote", out / "figure2.pdf", out / "figureS6.pdf")


if __name__ == "__main__":
    main()
