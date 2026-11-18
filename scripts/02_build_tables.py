"""Candidate generation C and signal extraction phi for the dev and test splits of one corpus.

python scripts/02_build_tables.py --config configs/gatelink.yaml --corpus bc5cdr --gold-spans

--gold-spans uses the gold mentions (the setting of the paper). Without it, mentions are predicted by a
recognizer trained with scripts/01_train_ner.py (optional end-to-end setting); --ner-model then selects it.
Writes runs/<...>/tables/<corpus>_<mode>_s<seed>_<split>.pkl and the per-document gold counts.
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

from gatelink.data import load_corpus
from gatelink.kb import load_kb
from gatelink.pipeline import build_table, gold_counts
from gatelink.retrieve import DenseRetriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gatelink.yaml")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gold-spans", action="store_true")
    ap.add_argument("--ner-model")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--tag", help="replaces the mode in output names, e.g. xfrom-ncbi for cross-corpus test tables")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    spec, rcfg = cfg["corpora"][a.corpus], cfg["retrieval"]
    out = Path(cfg["output_dir"]) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    mode = a.tag or ("gold" if a.gold_spans else "e2e")

    docs_by_split = {s: load_corpus(spec, s) for s in a.splits}
    types = sorted({m.etype for docs in docs_by_split.values() for d in docs for m in d.mentions} & set(cfg["kbs"]))
    kbs = {t: load_kb(cfg["kbs"][t]) for t in types}
    retrievers = {t: DenseRetriever(kbs[t], rcfg["model"], rcfg["device"], cache_dir=rcfg["cache_dir"]) for t in types}

    if not a.gold_spans:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        from gatelink.ner import predict
        path = a.ner_model or str(Path(cfg["output_dir"]) / "ner" / f"{a.corpus}_s{a.seed}" / "best")
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForTokenClassification.from_pretrained(path)
        labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
        from gatelink.device import best_device
        device = best_device(rcfg.get("device", "auto"))

    tie_keys = {}
    if spec.get("species_prior") and "Gene" in kbs:
        # species step for genes (zero-shot): species named in the document, then a fixed order, human first
        from gatelink.species import tie_breaker

    for split, docs in docs_by_split.items():
        if spec.get("species_prior") and "Gene" in kbs:
            tie_keys = {"Gene": tie_breaker({d.doc_id: d.text for d in docs}, kbs["Gene"].taxon)}
        for d in docs:                                   # evaluate only entity types with a KB
            d.mentions = [m for m in d.mentions if m.etype in types]
        if a.gold_spans:
            pred = [m for d in docs for m in d.mentions]
            pred = [type(m)(m.doc_id, m.start, m.end, m.text, m.etype) for m in pred]
        else:
            pred = [m for ms in predict(docs, model, tok, labels, device).values() for m in ms if m.etype in types]
        table = build_table(docs, pred, retrievers, rcfg["k"], kbs, tie_keys)
        stem = f"{a.corpus}_{mode}_s{a.seed}_{split}"
        table.to_pickle(out / f"{stem}.pkl")
        gold_counts(docs).to_csv(out / f"{stem}_gold.csv")
        print(stem, len(table), "mentions;", int(table["matched"].sum()) if len(table) else 0, "match gold spans")


if __name__ == "__main__":
    main()
