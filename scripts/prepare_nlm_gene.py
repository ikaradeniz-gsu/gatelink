"""Build train/dev/test BioC files for NLM-Gene from the Zenodo release.

The release has one BioC file per abstract (Corpus/FINAL/<PMID>.BioC.XML) and two PMID lists
(Pmidlist.Train.txt, Pmidlist.Test.txt). The official split has no development set, so a fixed random
sample of the training PMIDs (seed 0) is held out as dev.

python scripts/prepare_nlm_gene.py --src /tmp/nlmgene/Corpus --out data/corpora/nlm_gene --n-dev 50

It also prints the annotation types and infon keys found, so the config's type_map and id_keys can be checked.
"""
import argparse
import random
from collections import Counter
from pathlib import Path

from bioc import biocxml
from bioc.datastructure import BioCCollection


def read_list(p: Path) -> list[str]:
    return [l.strip() for l in p.read_text().split() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder holding FINAL/ and the Pmidlist files")
    ap.add_argument("--out", default="data/corpora/nlm_gene")
    ap.add_argument("--n-dev", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    lists = {p.name: p for p in src.rglob("Pmidlist*.txt")}
    train_l = next((p for n, p in lists.items() if "train" in n.lower()), None)
    test_l = next((p for n, p in lists.items() if "test" in n.lower()), None)
    files = {f.name.split(".")[0]: f for f in (src / "FINAL").glob("*.[Xx][Mm][Ll]")}
    print(f"{len(files)} abstracts in FINAL/; PMID lists: {sorted(lists)}")

    test = [p for p in read_list(test_l) if p in files] if test_l else []
    if train_l:
        train = [p for p in read_list(train_l) if p in files]
    else:                                    # no train list shipped: everything that is not test
        train = sorted(set(files) - set(test))
        print("no train list found: using every abstract not in the test list")
    rng = random.Random(a.seed)
    dev = sorted(rng.sample(sorted(train), a.n_dev))
    train = [p for p in train if p not in set(dev)]

    types, keys, n_ann = Counter(), Counter(), 0
    for split, pmids in (("train", train), ("dev", dev), ("test", test)):
        coll = BioCCollection()
        coll.source, coll.key = "NLM-Gene", "nlm_gene.key"
        for pmid in pmids:
            with open(files[pmid]) as fh:
                c = biocxml.load(fh)
            for d in c.documents:
                coll.add_document(d)
                for p in d.passages:
                    for ann in p.annotations:
                        n_ann += 1
                        types[ann.infons.get("type", "")] += 1
                        keys.update(k for k in ann.infons if k != "type")
        with open(out / f"{split}.BioC.xml", "w") as fh:
            biocxml.dump(coll, fh)
        print(f"{split}: {len(pmids)} abstracts -> {out / (split + '.BioC.xml')}")
    print(f"\n{n_ann:,} annotations; types: {types.most_common(8)}")
    print(f"infon keys: {keys.most_common(8)}")


if __name__ == "__main__":
    main()
