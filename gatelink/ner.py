"""Optional recognizer for the end-to-end setting with predicted mentions (not used in the paper, which
normalizes gold mentions). A BIO token classifier (PubMedBERT / BioLinkBERT) that also returns
span probability, boundary entropy and type probability for every predicted mention.

Training uses the HuggingFace Trainer. Documents are split into sentences so that every
window fits the encoder; offsets are kept so predictions map back to document characters.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from .schema import Document, Mention

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def sentences(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in _SENT.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def label_list(types: list[str]) -> list[str]:
    return ["O"] + [f"{p}-{t}" for t in types for p in ("B", "I")]


@dataclass
class Window:
    doc_id: str
    offset: int
    text: str


def windows(docs: list[Document]) -> list[Window]:
    return [Window(d.doc_id, s, d.text[s:e]) for d in docs for s, e in sentences(d.text)]


def encode_for_training(docs: list[Document], tokenizer, labels: list[str], max_len: int = 256):
    lab2id = {l: i for i, l in enumerate(labels)}
    feats = []
    for d in docs:
        char_lab = ["O"] * len(d.text)
        for m in d.mentions:
            for i in range(m.start, min(m.end, len(d.text))):
                char_lab[i] = ("B-" if i == m.start else "I-") + m.etype
        for s, e in sentences(d.text):
            enc = tokenizer(d.text[s:e], truncation=True, max_length=max_len, return_offsets_mapping=True)
            tags = []
            for (a, b) in enc["offset_mapping"]:
                if a == b:
                    tags.append(-100)                       # special tokens
                else:
                    tag = char_lab[s + a]
                    # a token that starts inside a mention but not at its first char is I-
                    if tag.startswith("B-") and a > 0 and char_lab[s + a - 1] != "O" \
                            and char_lab[s + a - 1][2:] == tag[2:] and not _is_start(d, s + a):
                        tag = "I-" + tag[2:]
                    tags.append(lab2id.get(tag, 0))
            enc.pop("offset_mapping")
            enc["labels"] = tags
            feats.append(enc)
    return feats


def _is_start(d: Document, pos: int) -> bool:
    return any(m.start == pos for m in d.mentions)


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1)
    return float(-(p * np.log(p)).sum() / math.log(len(p)))


def predict(docs: list[Document], model, tokenizer, labels: list[str], device="auto",
            batch_size: int = 32, max_len: int = 256) -> dict[str, list[Mention]]:
    """Returns doc_id -> predicted mentions with recognition-confidence signals."""
    import torch
    from .device import best_device

    device = best_device(device)
    model.eval().to(device)
    wins = windows(docs)
    out: dict[str, list[Mention]] = {d.doc_id: [] for d in docs}
    for i in range(0, len(wins), batch_size):
        batch = wins[i:i + batch_size]
        enc = tokenizer([w.text for w in batch], truncation=True, max_length=max_len, padding=True,
                        return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping").numpy()
        with torch.no_grad():
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
        probs = torch.softmax(logits, -1).cpu().numpy()
        for w, off, pr in zip(batch, offsets, probs):
            out[w.doc_id].extend(_decode(w, off, pr, labels))
    return out


def _decode(w: Window, offsets, probs, labels) -> list[Mention]:
    mentions, cur = [], None
    toks = [(j, a, b) for j, (a, b) in enumerate(offsets) if b > a]

    def close(end_idx):
        nonlocal cur
        if cur is None:
            return
        j0, j1, etype = cur["j0"], end_idx, cur["type"]
        a, b = offsets[j0][0], offsets[j1][1]
        tok_p = [probs[j].max() for j in range(j0, j1 + 1) if offsets[j][1] > offsets[j][0]]
        type_cols = [labels.index(f"B-{etype}"), labels.index(f"I-{etype}")]
        type_p = [probs[j][type_cols].sum() for j in range(j0, j1 + 1)]
        nxt = j1 + 1 if j1 + 1 < len(probs) else j1
        bent = (_entropy(probs[j0]) + _entropy(probs[nxt])) / 2
        mentions.append(Mention(w.doc_id, w.offset + int(a), w.offset + int(b), w.text[a:b], etype,
                                span_prob=float(np.mean(tok_p)), boundary_entropy=bent,
                                type_prob=float(np.mean(type_p))))
        cur = None

    prev = None
    for j, a, b in toks:
        lab = labels[int(probs[j].argmax())]
        if lab == "O":
            close(prev) if cur else None
        elif lab.startswith("B-") or cur is None or cur["type"] != lab[2:]:
            if cur:
                close(prev)
            cur = {"j0": j, "type": lab[2:]}
        prev = j
    if cur:
        close(prev)
    return mentions
