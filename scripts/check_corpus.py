"""Sanity checks for a new corpus, before any LLM call.

python scripts/check_corpus.py --config configs/gatelink.yaml --corpus bc5cdr            # stage 1: data + KB
python scripts/check_corpus.py --config configs/gatelink.yaml --corpus bc5cdr --tables   # stage 2: retrieval

Stage 1: documents and mentions per split and entity type, and how many gold ids exist in the KB
         (after mapping alternative ids). Coverage well below ~98% means a loader or KB-version problem.
Stage 2: after 02_build_tables.py --gold-spans: retriever accuracy@1 and recall@k per type, plus the
         top error pairs, so the S1 baseline can be checked against published zero-shot SapBERT numbers.
"""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from gatelink.data import load_corpus
from gatelink.kb import load_kb


def stage1(cfg, name):
    spec = cfg["corpora"][name]
    docs = {s: load_corpus(spec, s) for s in ("train", "dev", "test")}
    types = sorted({m.etype for ds in docs.values() for d in ds for m in d.mentions})
    print(f"\n{name}: documents and mentions with at least one id (mentions without ids are dropped)")
    for s, ds in docs.items():
        c = Counter(m.etype for d in ds for m in d.mentions)
        ids = [d.doc_id for d in ds]
        dup = len(ids) - len(set(ids))
        print(f"  {s:5s} {len(ds):5d} docs  " + "  ".join(f"{t} {c[t]:,}" for t in types)
              + (f"  (duplicate doc ids: {dup})" if dup else ""))
    for t in types:
        if t not in cfg["kbs"]:
            print(f"  {t}: no KB in the config; this type will be skipped")
            continue
        kb = load_kb(cfg["kbs"][t])
        print(f"\n  KB for {t}: {len(kb.preferred):,} concepts, "
              f"{sum(len(v) for v in kb.synonyms.values()):,} names, {len(kb.alt_to_primary):,} alternative ids")
        for s in ("dev", "test"):
            ms = [m for d in docs[s] for m in d.mentions if m.etype == t]
            ok = [any(g in kb.preferred or g in kb.alt_to_primary for g in m.gold_ids) for m in ms]
            missing = Counter(g for m, o in zip(ms, ok) if not o for g in m.gold_ids)
            print(f"  {s}: gold id in KB for {sum(ok):,} / {len(ms):,} = {100 * sum(ok) / max(len(ms), 1):.1f}%"
                  + (f"; most frequent missing: {missing.most_common(5)}" if missing else ""))


def stage2(cfg, name, k):
    tdir = Path(cfg["output_dir"]) / "tables"
    for s in ("dev", "test"):
        p = tdir / f"{name}_gold_s0_{s}.pkl"
        if not p.exists():
            print(f"  {p} not found: run 02_build_tables.py --gold-spans first")
            continue
        df = pd.read_pickle(p)
        df = df[df["matched"]]
        gold = df["gold_ids"].apply(set)
        df = df.assign(acc1=[r in g for r, g in zip(df["ret_id"], gold)],
                       rec=[any(c[0] in g for c in cs[:k]) for cs, g in zip(df["candidates"], gold)])
        print(f"\n{name} {s}: retriever accuracy@1 and recall@{k} on gold mentions")
        print((100 * df.groupby("etype")[["acc1", "rec"]].mean()).round(1)
              .assign(n=df.groupby("etype").size()).to_string())
        if s == "test":
            err = df[~df["acc1"]]
            print("  most frequent errors (mention -> retriever id, gold):")
            for (txt, r), n in Counter(zip(err["text"].str.lower(), err["ret_id"])).most_common(8):
                g = next(iter(err[(err["text"].str.lower() == txt) & (err["ret_id"] == r)]["gold_ids"]))
                print(f"    {n:3d}  {txt!r} -> {r}, gold {g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gatelink.yaml")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tables", action="store_true", help="stage 2: check the gold-span tables")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.tables:
        stage2(cfg, a.corpus, cfg["retrieval"]["k"])
    else:
        stage1(cfg, a.corpus)


if __name__ == "__main__":
    main()
