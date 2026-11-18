"""Optional: fine-tune the recognizer on one corpus, for the end-to-end setting with predicted mentions.
The paper uses gold mentions and does not need this step.

python scripts/01_train_ner.py --config configs/gatelink.yaml --corpus bc5cdr --seed 0
"""
import argparse
from pathlib import Path

import yaml

from gatelink.data import load_corpus
from gatelink.ner import encode_for_training, label_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gatelink.yaml")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    spec, ncfg = cfg["corpora"][a.corpus], cfg["ner"]

    from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                              Trainer, TrainingArguments, set_seed)
    set_seed(a.seed)
    train, dev = load_corpus(spec, "train"), load_corpus(spec, "dev")
    types = sorted({m.etype for d in train for m in d.mentions})
    labels = label_list(types)
    tok = AutoTokenizer.from_pretrained(ncfg["model"])
    model = AutoModelForTokenClassification.from_pretrained(
        ncfg["model"], num_labels=len(labels), id2label=dict(enumerate(labels)),
        label2id={l: i for i, l in enumerate(labels)})
    out = Path(cfg["output_dir"]) / "ner" / f"{a.corpus}_s{a.seed}"
    args = TrainingArguments(output_dir=str(out), num_train_epochs=ncfg["epochs"], learning_rate=ncfg["lr"],
                             per_device_train_batch_size=ncfg["batch_size"], eval_strategy="epoch",
                             save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="eval_loss",
                             save_total_limit=1, seed=a.seed, report_to=[], warmup_ratio=0.1, weight_decay=0.01)
    trainer = Trainer(model=model, args=args, tokenizer=tok,
                      train_dataset=encode_for_training(train, tok, labels, ncfg["max_len"]),
                      eval_dataset=encode_for_training(dev, tok, labels, ncfg["max_len"]),
                      data_collator=DataCollatorForTokenClassification(tok))
    trainer.train()
    trainer.save_model(str(out / "best"))
    tok.save_pretrained(str(out / "best"))
    print("saved", out / "best")


if __name__ == "__main__":
    main()
