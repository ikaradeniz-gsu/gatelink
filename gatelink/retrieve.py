"""Candidate generator C: abbreviation resolution + exact dictionary match + SapBERT dense retrieval.

Candidates are aggregated per concept (max over its synonyms) and the top-k concepts returned.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .kb import KB, norm_name
from .schema import Candidate, Mention

# ------------------------------------------------------------------ abbreviations (Schwartz & Hearst 2003)
_PAREN = re.compile(r"\(([^()]{1,10})\)")


def find_abbreviations(text: str) -> dict[str, str]:
    """Return short form -> long form pairs defined as 'long form (SF)' in the text."""
    pairs = {}
    for m in _PAREN.finditer(text):
        sf = m.group(1).strip()
        if not (2 <= len(sf) <= 10 and any(c.isalpha() for c in sf) and sf[0].isalnum()) or " " in sf.strip():
            continue
        words = text[:m.start()].split()
        window = words[-min(len(sf) + 5, len(sf) * 2):]
        lf = _best_long_form(sf, " ".join(window))
        if lf:
            pairs[sf] = lf
    return pairs


def _best_long_form(sf: str, lf: str) -> str | None:
    s, l = len(sf) - 1, len(lf) - 1
    sf_l, lf_l = sf.lower(), lf.lower()
    while s >= 0:
        c = sf_l[s]
        if not c.isalnum():
            s -= 1
            continue
        while l >= 0 and (lf_l[l] != c or (s == 0 and l > 0 and lf_l[l - 1].isalnum())):
            l -= 1
        if l < 0:
            return None
        l -= 1
        s -= 1
    start = lf.rfind(" ", 0, l + 1) + 1
    out = lf[start:].strip()
    return out if len(out) > len(sf) else None


# ------------------------------------------------------------------ dense retrieval
class DenseRetriever:
    def __init__(self, kb: KB, model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                 device="auto", batch_size=256, cache_dir: str | None = None):
        import torch
        from .device import best_device
        device = best_device(device)
        from transformers import AutoModel, AutoTokenizer

        self.kb, self.device, self.bs = kb, device, batch_size
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.torch = torch
        self.names = kb.name_table()
        # the cache key includes a hash of the name table, so a changed KB (new release, loader fix) is re-encoded
        import hashlib
        h = hashlib.sha1("\n".join(f"{n}\t{c}" for n, c in self.names).encode()).hexdigest()[:10]
        cache = Path(cache_dir) / f"{kb.name}.{model_name.split('/')[-1]}.{h}.npy" if cache_dir else None
        if cache and cache.exists():
            emb = np.load(cache)
        else:
            emb = self.encode([n for n, _ in self.names])
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, emb)
        # Exact inner-product search with torch (no FAISS: on macOS FAISS and PyTorch each load their own
        # OpenMP runtime and the process aborts). Half precision on GPU/MPS halves memory.
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        self.emb = torch.from_numpy(emb).to(device=device, dtype=dtype)

    def encode(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.bs):
            enc = self.tok(texts[i:i + self.bs], padding=True, truncation=True, max_length=25, return_tensors="pt")
            with self.torch.no_grad():
                cls = self.model(**{k: v.to(self.device) for k, v in enc.items()}).last_hidden_state[:, 0]
            cls = self.torch.nn.functional.normalize(cls, dim=-1)
            out.append(cls.cpu().numpy().astype("float32"))
        return np.concatenate(out) if out else np.zeros((0, 768), dtype="float32")

    def search(self, queries: list[str], k: int = 20, oversample: int = 5) -> list[list[Candidate]]:
        q = self.torch.from_numpy(self.encode(queries)).to(device=self.emb.device, dtype=self.emb.dtype)
        scores, idx = topk_inner_product(self.torch, q, self.emb, k * oversample)
        results = []
        for srow, irow in zip(scores, idx):
            best: dict[str, Candidate] = {}
            for s, i in zip(srow, irow):
                name, cid = self.names[i]
                if cid not in best or s > best[cid].score:
                    best[cid] = Candidate(cid, name, float(s))
            results.append(sorted(best.values(), key=lambda c: -c.score)[:k])
        return results


def topk_inner_product(torch, q, emb, k: int, chunk: int = 256):
    """Exact top-k by inner product (cosine for normalised vectors). Returns numpy (scores, indices)."""
    k = min(k, emb.shape[0])
    out_s, out_i = [], []
    for i in range(0, q.shape[0], chunk):
        sim = q[i:i + chunk] @ emb.T
        s, ix = torch.topk(sim.float(), k, dim=1)
        out_s.append(s.cpu().numpy())
        out_i.append(ix.cpu().numpy())
    if not out_s:
        return np.zeros((0, k)), np.zeros((0, k), dtype=int)
    return np.concatenate(out_s), np.concatenate(out_i)


def generate_candidates(mentions: list[Mention], doc_texts: dict[str, str], retriever: DenseRetriever,
                        k: int = 20, tie_key=None) -> tuple[list[list[Candidate]], list[str]]:
    """Returns candidates per mention and the query string actually used (after abbreviation expansion).
    tie_key(mention, candidate) -> tuple orders candidates with equal scores (e.g. the species step for
    genes); without it, ties keep the retriever's order."""
    abbrev = {d: find_abbreviations(t) for d, t in doc_texts.items()}
    queries = [abbrev[m.doc_id].get(m.text.strip(), m.text) for m in mentions]
    cands = retriever.search(queries, k=k * 5 if tie_key else k)
    exact = retriever.kb.exact_index()
    for m, q, cs in zip(mentions, queries, cands):
        # dictionary matches are forced into the list with score 1.0 (cosine upper bound)
        for cid in exact.get(norm_name(q), ()):
            if not any(c.concept_id == cid for c in cs):
                cs.insert(0, Candidate(cid, retriever.kb.preferred.get(cid, q), 1.0))
            else:
                for c in cs:
                    if c.concept_id == cid:
                        c.score = max(c.score, 1.0)
        if tie_key:
            cs.sort(key=lambda c: (-round(c.score, 4), *tie_key(m, c), c.concept_id))
        else:
            cs.sort(key=lambda c: -c.score)
        del cs[k:]
    return cands, queries
