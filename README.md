# GateLink

Code, data products and results for the paper

> İlknur Karadeniz. **GateLink: selective large language model reranking for biomedical entity normalization.**
> Submitted to *Bioinformatics*, 2026.

GateLink normalizes a biomedical mention (disease, chemical or gene) to a vocabulary identifier. A zero-shot
SapBERT retriever proposes candidates, and a calibrated gate sends a mention to an LLM reranker only when it
estimates that the retriever's top candidate is wrong. On four corpora and three open-weight Gemma 4 models,
always-on LLM reranking never significantly improved on the retriever, whereas GateLink did so in five of the
seven disease and chemical settings with 5–23% of the LLM calls.

```
document d, mention m ─► candidate generator C ─► signal extractor φ ─► gate g ─┬─ g ≥ τ ─► LLM reranker L ─► identifier or NIL
                         (exact match + SapBERT,   (features of Table 1)          │
                          top-20 concepts)                                        └─ g < τ ─► top candidate of C
```

The LLM is run once on every mention, and its answers are cached. Every system (S1–S9) is then evaluated
offline by choosing, for each mention, between the cached LLM answer and the retriever's top candidate, so all
systems are compared on identical LLM outputs.

## Contents

| Path | Content |
| --- | --- |
| `gatelink/` | Python package: corpus and vocabulary loaders, candidate generator, species step, features, gate and threshold rules, LLM reranker, evaluation |
| `scripts/` | Numbered pipeline steps (below), Figure 1, data preparation and checks |
| `configs/gatelink.yaml` | The configuration used for every experiment in the paper |
| `reproduce.sh` | Runs every experiment and builds the tables and figures |
| `llm_cache/` | Every LLM answer of the paper (raw text, token counts, latency), one JSONL file per model |
| `predictions/` | For every gold mention: gold identifiers, candidates and scores, gate features, retriever and LLM answers |
| `results/` | Output of `04_evaluate.py` for every corpus, reranker and cross-validation split (see `results/README.md`) |
| `tests/` | Tests that need no data, models or GPU |

## Installation

```bash
git clone https://github.com/ikaradeniz-gsu/gatelink.git && cd gatelink
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest -q tests/
```

The experiments were run on one Apple M5 Max workstation (36 GB unified memory) with Python 3.14.9,
PyTorch 2.15.0, Transformers 5.23.0, scikit-learn 1.9.1 and NumPy 2.5.4. PyTorch uses the Apple GPU (MPS),
an NVIDIA GPU (CUDA) or the CPU automatically.

## Quick start: the paper's tables and figures from the released results

No data, models or LLM are needed:

```bash
python scripts/05_paper_tables.py --results results --out paper_tables.md     # Tables 3-4 and the Supplementary Tables
python scripts/06_figures.py --results results --out figures                  # Figure 2 and the calibration figure (Supplementary Section S6)
python scripts/figure1_overview.py figures/figure1                            # Figure 1
```

To re-train the gates and re-evaluate every system from the released per-mention tables (no vocabularies,
SapBERT or LLM needed; about 40 s per run on a laptop):

```bash
python scripts/release_tables.py import --src predictions        # -> runs/gatelink/tables/
python scripts/04_evaluate.py --corpus bc5cdr --llm mid --seed 0  # -> runs/gatelink/results/bc5cdr_gold_mid_s0/
```

## Full reproduction

### Data

Place the files under `data/` with the paths in `configs/gatelink.yaml`.

| Corpus | Entity type | Source |
| --- | --- | --- |
| NCBI Disease | Disease | https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/ (`NCBItrainset_corpus.txt`, `NCBIdevelopset_corpus.txt`, `NCBItestset_corpus.txt`) |
| BC5CDR | Disease, Chemical | http://www.biocreative.org/tasks/biocreative-v/track-3-cdr/ (`CDR_TrainingSet.PubTator.txt`, `CDR_DevelopmentSet.PubTator.txt`, `CDR_TestSet.PubTator.txt`) |
| NLM-Chem | Chemical | doi:10.5061/dryad.3tx95x6dz or Zenodo record 4628233 (`BC7T2-NLMChem-corpus-{train,dev,test}.BioC.xml`) |
| NLM-Gene | Gene | doi:10.5061/dryad.dv41ns1wt or Zenodo record 5089049; build the splits with `python scripts/prepare_nlm_gene.py --src <release>/Corpus --out data/corpora/nlm_gene --n-dev 50` (NLM-Gene has no development split, so 50 training documents, chosen with seed 0, form the development set) |

| Vocabulary | File | Source | Version used |
| --- | --- | --- | --- |
| CTD MEDIC (diseases) | `data/kb/CTD_diseases.tsv.gz` | https://ctdbase.org/downloads/ (free for non-commercial use) | downloaded 15 November 2026 |
| CTD chemicals | `data/kb/CTD_chemicals.tsv.gz` | https://ctdbase.org/downloads/ | downloaded 16 November 2026 |
| NCBI Gene | `data/kb/gene_info.gz` | https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz (filtered to 12 taxa when loaded) | downloaded 17 November 2026 |

The retriever (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`) is downloaded from Hugging Face on first use.
Vocabularies change between releases; newer releases can change candidate lists and therefore results.
`python scripts/check_corpus.py --corpus bc5cdr` checks that a corpus and its vocabulary load correctly.

### LLM rerankers

The rerankers are served locally with [Ollama](https://ollama.com):

```bash
OLLAMA_NUM_PARALLEL=4 OLLAMA_CONTEXT_LENGTH=4096 ollama serve
ollama pull gemma4:12b && ollama pull gemma4:26b && ollama pull gemma4:31b
```

The prompt is in `gatelink/llm_rerank.py`; reasoning is disabled, the temperature is 0 and the output is
limited to 8 tokens (32 for gemma4:31b). With the released `llm_cache/`, no LLM server is needed and every
answer is identical to the paper's. A cached answer is a candidate number, so it refers to the candidate list
of the paper; use the vocabulary versions listed above, or `predictions/` (see Quick start), to keep the two
consistent.

### Run

```bash
bash reproduce.sh                         # uses the released LLM answers
USE_RELEASED_CACHE=0 bash reproduce.sh    # queries the LLMs through Ollama
```

| Step | Script | Output (under `runs/gatelink/`) |
| --- | --- | --- |
| 1. Candidates C and features φ | `scripts/02_build_tables.py --corpus <c> --gold-spans` | `tables/<c>_gold_s0_{dev,test}.pkl` |
| 2. LLM reranker L on every mention | `scripts/03_run_llm.py --table <table> --llm {small,mid,large}` | `tables/..._{dev,test}__<llm>.pkl`, `llm_cache/<llm>.jsonl` |
| 3. Gate g, thresholds, systems S1–S9 | `scripts/04_evaluate.py --corpus <c> --llm <llm> --seed {0,1,2}` | `results/<c>_gold_<llm>_s<seed>/` |
| 4. Tables | `scripts/05_paper_tables.py` | `paper_tables.md` |
| 5. Figures | `scripts/06_figures.py`, `scripts/figure1_overview.py` | `figures/` |

`small`, `mid` and `large` are gemma4:12b, gemma4:26b and gemma4:31b. `--seed` selects one of the three
cross-validation splits of the development set used to set τ. `scripts/final_counts.py` counts invalid LLM
answers and the overlap of mention strings between development and test sets.

### Where each item of the paper comes from

| Paper | Source |
| --- | --- |
| Tables 3 and 4 | `results/*/main.csv` → `05_paper_tables.py` |
| Figure 2 | `results/<c>_gold_mid_s0/curves.csv` → `06_figures.py` |
| Section 3.3 (gate quality) | `results/*/summary.json`, `matched_budget.csv`, `auc_fc.csv`, `ablation.csv` |
| Section 3.4 (error analysis) | `results/*/per_type.csv`, `predictions/` |
| Supplementary Tables S3–S5, S7, S9, S11, S12 | `05_paper_tables.py` |
| Supplementary Section S6 (calibration figure) | `results/<c>_gold_mid_s0/calibration.csv` → `06_figures.py` |

## Method summary

* **Candidate generator C.** Schwartz–Hearst abbreviation expansion; exact match on normalized names (score 1)
  and zero-shot SapBERT over all vocabulary names; the score of a concept is the maximum over its names; top 20.
  For genes, equally scored candidates are ordered by how often the document names each species, then by a
  fixed species order with human first (`gatelink/species.py`).
* **Signal extractor φ.** 14 features in four groups (retrieval, lexical, ontology, context) plus the entity
  type (`gatelink/features.py`). Three recognition features are computed for predicted mentions only; they are
  constant on gold mentions.
* **Gate g.** Label z(m) = 1 if the retriever's top candidate is wrong. HistGradientBoosting with isotonic
  calibration; out-of-fold scores from five folds grouped by document (`gatelink/gate.py`). The gate is trained
  without LLM answers.
* **Threshold rules.** A: the largest τ whose development accuracy is within δ = 0.5 points of the best.
  B: a fixed share of 30% of mentions. R: conformal risk control, which bounds the expected share of harmful
  skips (mentions not sent although the LLM would have corrected the retriever) by ε = 1%.
* **Evaluation.** Accuracy@1 on gold mentions (NIL counts as wrong); two-sided paired bootstrap over
  documents with 10,000 resamples; a difference is significant only if p < 0.05 in all three splits.

The package also supports predicted mentions (`scripts/01_train_ner.py`, `02_build_tables.py` without
`--gold-spans`, metric: strict micro-F1); this setting is not used in the paper.

## Citation

```bibtex
@article{karadeniz2026gatelink,
  author  = {Karadeniz, {\.I}lknur},
  title   = {{GateLink}: selective large language model reranking for biomedical entity normalization},
  journal = {Bioinformatics},
  note    = {Submitted},
  year    = {2026}
}
```

## License

Code: MIT (see `LICENSE`). The corpora and vocabularies keep their own licenses; CTD data are free for
non-commercial use.
