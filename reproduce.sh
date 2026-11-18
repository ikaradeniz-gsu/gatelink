#!/usr/bin/env bash
# Reproduce every experiment of the paper (gold mentions), then build its tables and figures.
#
# Run from the repository root after the data are in place (README.md, "Data"):
#   bash reproduce.sh
#
# By default the released LLM answers (llm_cache/) are copied into the run folder first, so no LLM server
# is needed and every answer is identical to the paper's. To query the models yourself, start Ollama
# (README.md, "LLM rerankers") and run:
#   USE_RELEASED_CACHE=0 bash reproduce.sh
set -euo pipefail
CFG=${CFG:-configs/gatelink.yaml}
PY=${PYTHON:-python}
OUT=$($PY -c "import yaml; print(yaml.safe_load(open('$CFG'))['output_dir'])")
T=$OUT/tables

if [ "${USE_RELEASED_CACHE:-1}" = "1" ]; then
  mkdir -p "$OUT/llm_cache"
  cp -n llm_cache/*.jsonl "$OUT/llm_cache/"
fi

# 1. Candidate generation C and signal extraction phi on gold mentions (dev and test splits)
for corpus in ncbi_disease bc5cdr nlm_chem nlm_gene_sp; do
  $PY scripts/02_build_tables.py --config "$CFG" --corpus "$corpus" --gold-spans
done

# 2. LLM reranker L, once per mention (always-on); gated systems are evaluated offline from these answers
PAIRS="ncbi_disease:small ncbi_disease:mid ncbi_disease:large
bc5cdr:small bc5cdr:mid
nlm_chem:small nlm_chem:mid
nlm_gene_sp:small nlm_gene_sp:mid"
for p in $PAIRS; do
  for split in dev test; do
    $PY scripts/03_run_llm.py --config "$CFG" --table "$T/${p%%:*}_gold_s0_${split}.pkl" --llm "${p##*:}"
  done
done

# 3. Gate training, threshold rules and evaluation of S1-S9 on three cross-validation splits
for seed in 0 1 2; do
  for p in $PAIRS; do
    $PY scripts/04_evaluate.py --config "$CFG" --corpus "${p%%:*}" --llm "${p##*:}" --seed "$seed"
  done
done

# 4. Transfer between corpora (Supplementary Table S7): gate and threshold from another corpus's dev set
for llm in small mid; do
  $PY scripts/04_evaluate.py --config "$CFG" --corpus bc5cdr   --dev-corpus ncbi_disease --etype Disease  --llm "$llm" --seed 0
  $PY scripts/04_evaluate.py --config "$CFG" --corpus nlm_chem --dev-corpus bc5cdr       --etype Chemical --llm "$llm" --seed 0
done

# 5. Tables, figures and counts
$PY scripts/05_paper_tables.py --results "$OUT/results" --out "$OUT/paper_tables.md"
$PY scripts/06_figures.py --results "$OUT/results" --out "$OUT/figures"
$PY scripts/figure1_overview.py "$OUT/figures/figure1"
$PY scripts/final_counts.py --config "$CFG"
echo "done: $OUT/paper_tables.md and $OUT/figures/"
