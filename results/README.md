# Results

One folder per run of `scripts/04_evaluate.py`, named `<corpus>_gold_<reranker>_s<split>`:
`<reranker>` is `small` (gemma4:12b), `mid` (gemma4:26b) or `large` (gemma4:31b), and `<split>` (0, 1, 2) is the
cross-validation split of the development set used to set the threshold. `<source>-to-<target>_gold_...` folders
hold the transfer runs (gate and threshold from the source corpus, applied to the target corpus).

| File | Content |
| --- | --- |
| `main.csv` | Systems S1–S9: accuracy@1 (`acc1`), precision/recall/F1 with NIL as abstention, NIL share, share of mentions sent to the LLM, tokens, latency, harmful and beneficial skips, differences and bootstrap p-values against S1 and S2 |
| `summary.json` | Thresholds τ, gate AUROC and expected calibration error on test, random gate (mean and SD over 20 draws) |
| `curves.csv` | Accuracy@1 against the share of mentions sent to the LLM, for each gate (Figure 2; the column `F1` holds accuracy@1) |
| `auc_fc.csv` | Area under each curve |
| `matched_budget.csv` | GateLink against the training-free gates at 2.5–30% of mentions |
| `per_type.csv` | Results per entity type |
| `ablation.csv` | Gate AUROC and calibration error without each feature group (out-of-fold development scores) |
| `conformal.csv` | Risk control: target ε against realized share of harmful skips |
| `calibration.csv` | Gate probability and label for every test mention |
| `scores.csv` | Gate scores of GateLink and the training-free gates for every test mention |
| `random_draws.csv` | Accuracy curves of the 20 random-gate draws |

`nlm_gene_gold_*` (NLM-Gene **without** the species step, Supplementary Table S12) come from an earlier run that
grouped the gate's cross-validation folds by mention rather than by document. They are kept for transparency;
the current code (`--corpus nlm_gene`) groups folds by document and will give slightly different numbers.
All other folders are reproduced exactly by the code in this repository.
