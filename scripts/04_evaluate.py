"""Train the gate on the development mentions and evaluate systems S1-S9 on the test mentions.

python scripts/04_evaluate.py --config configs/gatelink.yaml --corpus bc5cdr --llm mid --seed 0

--seed selects the cross-validation split of the development set (0, 1, 2 in the paper).
Transfer between corpora (Supplementary Table S7): gate and threshold from another corpus's dev set,
  --corpus bc5cdr --dev-corpus ncbi_disease --etype Disease
Writes <output_dir>/results/<corpus>_gold_<llm>_s<seed>/ (main.csv, curves.csv, summary.json, ...).
"""
import argparse
from pathlib import Path

import yaml

from gatelink.experiment import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gatelink.yaml")
    ap.add_argument("--corpus", required=True, help="test corpus")
    ap.add_argument("--dev-corpus", help="corpus whose dev split trains the gate (default: same corpus)")
    ap.add_argument("--etype", help="restrict to one entity type (cross-corpus runs)")
    ap.add_argument("--mode", default="gold", help="gold (gold mentions, the paper) | e2e (predicted mentions)")
    ap.add_argument("--test-mode", help="tag of the test table if different, e.g. xfrom-ncbi")
    ap.add_argument("--llm", required=True)
    ap.add_argument("--seed", type=int, default=0, help="seed of the gate (and of the tagger tables in e2e mode)")
    ap.add_argument("--table-seed", type=int, help="seed of the tables to read; default: 0 in gold mode "
                    "(no tagger, tables are identical across seeds), else --seed")
    ap.add_argument("--suffix", default="", help="e.g. _n200 when 03_run_llm.py used --limit-docs 200")
    ap.add_argument("--bootstrap", type=int, default=10000)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    tables = Path(cfg["output_dir"]) / "tables"
    dev_c = a.dev_corpus or a.corpus
    ts = a.table_seed if a.table_seed is not None else (0 if a.mode == "gold" else a.seed)
    dev = tables / f"{dev_c}_{a.mode}_s{ts}_dev__{a.llm}{a.suffix}.pkl"
    test = tables / f"{a.corpus}_{a.test_mode or a.mode}_s{ts}_test__{a.llm}{a.suffix}.pkl"
    tag = f"{dev_c}-to-{a.corpus}" if a.dev_corpus else a.corpus
    out = Path(cfg["output_dir"]) / "results" / f"{tag}_{a.mode}_{a.llm}_s{a.seed}"
    metric = "acc1" if a.mode == "gold" else "f1"
    lcfg = next((x for x in cfg.get("llm", []) if x["name"] == a.llm), {})
    summary = run(dev, test, cfg["gate"], out, a.seed, bootstrap=a.bootstrap, only_etype=a.etype, metric=metric,
                  active_params_b=lcfg.get("active_params_b"))
    print(out, summary)


if __name__ == "__main__":
    main()
