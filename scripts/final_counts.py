"""Invalid LLM answers per corpus and reranker (Supplementary Table S10) and the share of test mention strings
that also occur in the development set (Section 2.4).

python scripts/final_counts.py --config configs/gatelink.yaml
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

CORPORA = ["ncbi_disease", "bc5cdr", "nlm_chem", "nlm_gene_sp"]
LLMS = ["small", "mid", "large"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gatelink.yaml")
    a = ap.parse_args()
    tables = Path(yaml.safe_load(open(a.config))["output_dir"]) / "tables"

    print("== invalid answers (dev / test)")
    for c in CORPORA:
        for llm in LLMS:
            out = []
            for split in ["dev", "test"]:
                p = tables / f"{c}_gold_s0_{split}__{llm}.pkl"
                if p.exists():
                    t = pd.read_pickle(p)
                    out.append(f"{int((~t['llm_valid']).sum())} of {len(t)}")
            if out:
                print(c, llm, " / ".join(out))

    print("== test mention strings also seen in dev (%)")
    for c in CORPORA:
        d = pd.read_pickle(tables / f"{c}_gold_s0_dev.pkl")
        t = pd.read_pickle(tables / f"{c}_gold_s0_test.pkl")
        seen = set(d["text"].str.lower())
        print(c, round(100 * t["text"].str.lower().isin(seen).mean(), 1))


if __name__ == "__main__":
    main()
