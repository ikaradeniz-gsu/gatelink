"""LLM reranker L: multiple-choice selection over the top-k candidates, served by Ollama (native API, used
in the paper) or any OpenAI-compatible chat endpoint (e.g. vLLM).

Every call is cached in a JSONL file keyed by mention uid, with token counts and latency, so the
LLM is run ONCE per mention (the always-on setting) and every gated system is evaluated offline
by choosing, per mention, between the cached LLM answer and the retriever's top-1.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .schema import NIL, Candidate, Mention

SYSTEM = ("You are an expert biomedical curator. You link a mention in a biomedical text to the single "
          "concept it refers to. Answer with the number of the correct option only.")

TEMPLATE = """Text: {context}

Mention: "{mention}" (entity type: {etype})

Which concept does the mention refer to?
{options}
0. None of the above, or the mention is not a {etype}

Answer with one number."""


def build_prompt(m: Mention, context: str, cands: list[Candidate], kb, max_syn: int = 3) -> str:
    opts = []
    for i, c in enumerate(cands, 1):
        syns = [s for s in kb.synonyms.get(c.concept_id, []) if s != kb.preferred.get(c.concept_id)][:max_syn]
        info = kb.info.get(c.concept_id, "")
        line = f"{i}. {kb.preferred.get(c.concept_id, c.name)}"
        if syns:
            line += f" (synonyms: {'; '.join(syns)})"
        if info:
            line += f" [{info[:120]}]"
        opts.append(line)
    return TEMPLATE.format(context=context, mention=m.text, etype=m.etype, options="\n".join(opts))


def context_window(text: str, start: int, end: int, width: int = 300) -> str:
    a, b = max(0, start - width), min(len(text), end + width)
    return ("..." if a else "") + text[a:start] + "[[" + text[start:end] + "]]" + text[end:b] + ("..." if b < len(text) else "")


def parse_answer(raw: str, cands: list[Candidate]) -> tuple[str, bool]:
    """Returns (concept id or NIL, valid?). Reads the LAST standalone integer in the reply (word
    boundaries, so "WT1" or "HPT-JT" never count); for a one-number reply this is that number.
    0 = NIL; 1..k = that candidate; no integer, or one above k = invalid (falls back to the retriever)."""
    nums = re.findall(r"(?<![\w.-])\d+(?![\w-])(?!\.\d)", raw or "")
    if not nums:
        return NIL, False
    i = int(nums[-1])
    if i == 0:
        return NIL, True
    if 1 <= i <= len(cands):
        return cands[i - 1].concept_id, True
    return NIL, False


class LLMReranker:
    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY",
                 cache: str | Path = "cache/llm.jsonl", max_workers: int = 16, temperature: float = 0.0,
                 n_gpus: int = 1, api: str = "openai", num_ctx: int = 4096, think: bool = False,
                 max_tokens: int = 8):
        """api="openai": any OpenAI-compatible server (vLLM, LM Studio).
        api="ollama": Ollama's native /api/chat, which can switch reasoning off (think=False); needed for
        models such as Gemma 4 or Qwen3 that otherwise spend the answer budget on hidden reasoning."""
        self.api, self.num_ctx, self.think, self.max_tokens = api, num_ctx, think, max_tokens
        if api == "ollama":
            self.base = base_url.rstrip("/").removesuffix("/v1")
        else:
            from openai import OpenAI
            self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model, self.temperature, self.max_workers, self.n_gpus = model, temperature, max_workers, n_gpus
        self.cache_path = Path(cache)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.open():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:                  # a line cut off by an interrupted run
                    continue
                self.cache[r["uid"]] = r

    def _call_ollama(self, uid: str, prompt: str) -> dict:
        import urllib.request
        body = {"model": self.model, "stream": False, "think": self.think,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                "options": {"temperature": self.temperature, "num_predict": self.max_tokens, "num_ctx": self.num_ctx, "seed": 0}}
        req = urllib.request.Request(self.base + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(r.read())
        dt = time.perf_counter() - t0
        return {"uid": uid, "raw": out.get("message", {}).get("content", ""), "latency_s": dt,
                "prompt_tokens": out.get("prompt_eval_count"), "completion_tokens": out.get("eval_count"),
                "model": self.model}

    def _call(self, uid: str, prompt: str) -> dict:
        if self.api == "ollama":
            return self._call_ollama(uid, prompt)
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
        dt = time.perf_counter() - t0
        u = resp.usage
        return {"uid": uid, "raw": resp.choices[0].message.content, "latency_s": dt,
                "prompt_tokens": getattr(u, "prompt_tokens", None),
                "completion_tokens": getattr(u, "completion_tokens", None), "model": self.model}

    def run(self, items: list[tuple[str, str]], progress_every: int = 200) -> dict[str, dict]:
        """items: (uid, prompt). Returns uid -> record (from cache when available).
        Each answer is appended to the cache as soon as it arrives, so an interrupted run loses nothing:
        running the same command again sends only the mentions that are not cached yet."""
        import sys
        from concurrent.futures import as_completed
        todo = [(u, p) for u, p in items if u not in self.cache]
        if len(todo) < len(items):
            print(f"{len(items) - len(todo):,} of {len(items):,} mentions already cached; {len(todo):,} to send",
                  file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        new = []
        with ThreadPoolExecutor(self.max_workers) as ex, self.cache_path.open("a") as fh:
            futs = [ex.submit(self._call, u, p) for u, p in todo]
            try:
                done = enumerate(as_completed(futs), 1)
                self._consume(done, fh, new, len(todo), t0, progress_every)
            except BaseException:                             # Ctrl-C or an error: stop queued calls, keep answers
                for f in futs:
                    f.cancel()
                print(f"stopped; {len(new):,} new answers are saved in {self.cache_path}", file=sys.stderr)
                raise
        self._finish(new, t0)
        return {u: self.cache[u] for u, _ in items}

    def _consume(self, done, fh, new, n_todo, t0, progress_every):
        import sys
        for i, f in done:
            rec = f.result()
            self.cache[rec["uid"]] = rec
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            new.append(rec)
            if i % progress_every == 0 or i == n_todo:
                el = time.perf_counter() - t0
                eta = el / i * (n_todo - i)
                print(f"  {i:,}/{n_todo:,} answered  {i / el:.2f}/s  elapsed {el / 60:.0f} min  "
                      f"left ~{eta / 60:.0f} min", file=sys.stderr, flush=True)

    def _finish(self, new, t0):
        # GPU-seconds amortised over the calls of this run: wall-clock x GPUs serving the model.
        # Run on a dedicated server so that this is the model's own compute.
        if new:
            gpu_s = (time.perf_counter() - t0) * self.n_gpus / len(new)
            for rec in new:
                rec["gpu_s"] = gpu_s
            tmp = self.cache_path.with_suffix(".tmp")           # rewrite once so the file carries gpu_s too
            with tmp.open("w") as fh:
                for rec in self.cache.values():
                    fh.write(json.dumps(rec) + "\n")
            tmp.replace(self.cache_path)
