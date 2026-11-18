"""Export or import the per-mention answer tables released in predictions/.

Each released file holds, for every gold mention of one corpus split and one reranker: the mention, its gold
identifiers, the top-20 candidate identifiers with retrieval scores, the gate features of Table 1, the
retriever's answer and the LLM's answer. Candidate names are not included (they come from the vocabularies,
whose licenses are listed in README.md).

Import (rebuilds the tables that 04_evaluate.py reads, so the gate and all systems can be evaluated without
the vocabularies, SapBERT or an LLM):
    python scripts/release_tables.py import --config configs/gatelink.yaml --src predictions

Export (how predictions/ was made from the tables of 03_run_llm.py):
    python scripts/release_tables.py export --config configs/gatelink.yaml --dst predictions
"""
import argparse
import gzip
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

DROP = ["query"]                      # recomputable; not needed for evaluation


def _plain(v):
    """numpy scalars and lists -> plain Python values for json."""
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def export(tables: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(tables.glob("*_gold_s0_*__*.pkl")):
        if p.stem.endswith(("_n20",)):                  # pilot runs on 20 documents
            continue
        t = pd.read_pickle(p).drop(columns=DROP, errors="ignore")
        t["candidates"] = t["candidates"].map(lambda cs: [[c[0], float(c[2])] for c in cs])
        # the json module writes floats with repr(), so every value is read back bit for bit
        with gzip.open(dst / f"{p.stem}.jsonl.gz", "wt", encoding="utf-8") as fh:
            for rec in t.to_dict(orient="records"):
                fh.write(json.dumps({k: _plain(v) for k, v in rec.items()}, ensure_ascii=False) + "\n")
        shutil.copy(p.with_name(p.stem + "_gold.csv"), dst / f"{p.stem}_gold.csv")
        print("exported", p.stem, len(t))


def import_(src: Path, tables: Path):
    tables.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.glob("*.jsonl.gz")):
        stem = p.name.removesuffix(".jsonl.gz")
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            t = pd.DataFrame([json.loads(line) for line in fh])
        t["candidates"] = t["candidates"].map(lambda cs: [[c[0], "", c[1]] for c in cs])
        t.to_pickle(tables / f"{stem}.pkl")
        shutil.copy(src / f"{stem}_gold.csv", tables / f"{stem}_gold.csv")
        print("imported", stem, len(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["export", "import"])
    ap.add_argument("--config", default="configs/gatelink.yaml")
    ap.add_argument("--src", default="predictions")
    ap.add_argument("--dst", default="predictions")
    a = ap.parse_args()
    tables = Path(yaml.safe_load(open(a.config))["output_dir"]) / "tables"
    if a.action == "export":
        export(tables, Path(a.dst))
    else:
        import_(Path(a.src), tables)


if __name__ == "__main__":
    main()
