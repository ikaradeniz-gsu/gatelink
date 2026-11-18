"""Run the LLM reranker ONCE on every mention of a table (always-on setting) and store answers,
tokens and latency. Gated systems are then evaluated offline from these cached answers.

python scripts/03_run_llm.py --config configs/gatelink.yaml --table runs/gatelink/tables/bc5cdr_gold_s0_test.pkl --llm mid

Answers are cached in <output_dir>/llm_cache/<llm>.jsonl; mentions already in the cache are not sent again.
With the released caches (llm_cache/ in the repository), this step needs no LLM server.

Clean timing run for the paper (after the normal run): re-runs every call with a fresh cache after a warm-up,
then writes the measured tokens/latency/accelerator-seconds into the same output table (answers are kept from
the normal run) and saves a JSON with the run's environment and throughput:
python scripts/03_run_llm.py ... --timing-run clean1
"""
import argparse
from pathlib import Path

import yaml

from gatelink.data import load_corpus
from gatelink.kb import load_kb
from gatelink.llm_rerank import LLMReranker, build_prompt, context_window, parse_answer
from gatelink.schema import Candidate, Mention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gatelink.yaml")
    ap.add_argument("--table", required=True)
    ap.add_argument("--llm", required=True, help="name of an entry under llm: in the config")
    ap.add_argument("--limit-docs", type=int, help="sample this many documents (fixed seed) to cap cost")
    ap.add_argument("--retry-invalid", action="store_true", help="re-send cached answers that do not parse")
    ap.add_argument("--timing-run", metavar="NAME", help="fresh-cache, warmed-up timing run; updates cost columns only")
    ap.add_argument("--warmup", type=int, default=3, help="untimed warm-up calls before a timing run")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    lcfg = next(x for x in cfg["llm"] if x["name"] == a.llm)

    import pandas as pd
    table = pd.read_pickle(a.table)
    gold_csv = Path(a.table).with_name(Path(a.table).stem + "_gold.csv")
    suffix = f"__{a.llm}"
    if a.limit_docs:
        gold = pd.read_csv(gold_csv, index_col=0)
        keep = gold.sample(min(a.limit_docs, len(gold)), random_state=0).index.astype(str)
        table = table[table["doc_id"].astype(str).isin(set(keep))]
        suffix += f"_n{a.limit_docs}"
        gold.loc[gold.index.astype(str).isin(set(keep))].to_csv(
            Path(a.table).with_name(Path(a.table).stem + suffix + "_gold.csv"))
    split = Path(a.table).stem.split("_")[-1]
    corpus = max((c for c in cfg["corpora"] if Path(a.table).stem.startswith(c + "_")), key=len)  # longest match
    texts = {d.doc_id: d.text for d in load_corpus(cfg["corpora"][corpus], split)}
    kbs = {t: load_kb(cfg["kbs"][t]) for t in table["etype"].unique()}

    items, cands_by_uid = [], {}
    for r in table.itertuples():
        cands = [Candidate(c[0], c[1], c[2]) for c in r.candidates]
        cands_by_uid[f"{corpus}|{r.uid}"] = cands
        m = Mention(r.doc_id, r.start, r.end, r.text, r.etype)
        prompt = build_prompt(m, context_window(texts[r.doc_id], r.start, r.end), cands, kbs[r.etype])
        items.append((f"{corpus}|{r.uid}", prompt))   # corpus in the key: PMIDs recur across corpora

    cache = Path(cfg["output_dir"]) / "llm_cache" / f"{a.llm}.jsonl"
    if a.timing_run:
        return timing_run(a, cfg, lcfg, table, items, corpus, split, suffix)
    rr = LLMReranker(lcfg["model"], lcfg["base_url"], lcfg.get("api_key", "EMPTY"), cache, lcfg.get("max_workers", 16),
                      n_gpus=lcfg.get("n_gpus", 1), api=lcfg.get("api", "openai"),
                      num_ctx=lcfg.get("num_ctx", 4096), think=lcfg.get("think", False),
                      max_tokens=lcfg.get("max_tokens", 8))
    if a.retry_invalid:
        # re-send only cached answers that do not parse; the rest of the cache is kept as it is
        bad = {u for u, p in items if u in rr.cache and not parse_answer(rr.cache[u]["raw"], cands_by_uid[u])[1]}
        for u in bad:
            del rr.cache[u]
        print(f"re-sending {len(bad)} unparseable answers with max_tokens={rr.max_tokens}")
    recs = rr.run(items)
    parsed = {u: parse_answer(recs[u]["raw"], cands_by_uid[u]) for u in recs}
    key = table["uid"].map(lambda u: f"{corpus}|{u}")
    table["llm_id"] = key.map(lambda u: parsed[u][0])
    table["llm_valid"] = key.map(lambda u: parsed[u][1])
    table["llm_prompt_tokens"] = key.map(lambda u: recs[u]["prompt_tokens"])
    table["llm_completion_tokens"] = key.map(lambda u: recs[u]["completion_tokens"])
    table["llm_latency_s"] = key.map(lambda u: recs[u]["latency_s"])
    table["llm_gpu_s"] = key.map(lambda u: recs[u].get("gpu_s"))
    out = Path(a.table).with_name(Path(a.table).stem + suffix + ".pkl")
    if not a.limit_docs:
        import shutil
        shutil.copy(gold_csv, Path(a.table).with_name(Path(a.table).stem + suffix + "_gold.csv"))
    table.to_pickle(out)
    print("wrote", out, "| invalid answers:", int((~table["llm_valid"]).sum()))


def environment(lcfg: dict) -> dict:
    """Machine and software facts to report with timings (best effort on macOS/Linux)."""
    import platform
    import subprocess
    import sys

    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return None
    model_line = next((l for l in (sh(["ollama", "list"]) or "").splitlines()
                       if l.split() and l.split()[0] == lcfg["model"]), None) if lcfg.get("api") == "ollama" else None
    return {"os": platform.platform(), "macos": platform.mac_ver()[0] or None,
            "cpu": sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "memory_gb": round(int(sh(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30, 1) or None,
            "power": sh(["pmset", "-g", "batt"]), "python": sys.version.split()[0],
            "ollama_version": sh(["ollama", "--version"]) if lcfg.get("api") == "ollama" else None,
            "model": lcfg["model"], "model_list_entry": model_line,
            "max_workers": lcfg.get("max_workers"), "n_gpus": lcfg.get("n_gpus", 1)}


def timing_run(a, cfg, lcfg, table, items, corpus, split, suffix):
    import json
    import time

    import numpy as np
    import pandas as pd

    env = environment(lcfg)
    if env["power"] and "AC Power" not in env["power"]:
        print("WARNING: not on AC power; plug in the Mac for timing runs")
    tdir = Path(cfg["output_dir"]) / "timing"
    tdir.mkdir(parents=True, exist_ok=True)
    cache = tdir / f"{a.llm}__{corpus}_{split}__{a.timing_run}.jsonl"
    if cache.exists():
        raise SystemExit(f"{cache} exists: choose a new --timing-run name so the run starts from an empty cache")
    rr = LLMReranker(lcfg["model"], lcfg["base_url"], lcfg.get("api_key", "EMPTY"), cache, lcfg.get("max_workers", 16),
                     n_gpus=lcfg.get("n_gpus", 1), api=lcfg.get("api", "openai"),
                     num_ctx=lcfg.get("num_ctx", 4096), think=lcfg.get("think", False),
                     max_tokens=lcfg.get("max_tokens", 8))
    for i in range(min(a.warmup, len(items))):                # load the model and fill caches, untimed
        rr._call("warmup", items[i][1])
    t0 = time.perf_counter()
    recs = rr.run(items)
    wall = time.perf_counter() - t0

    out = Path(a.table).with_name(Path(a.table).stem + suffix + ".pkl")
    lat = np.array([recs[u]["latency_s"] for u, _ in items])
    tok = np.array([(recs[u]["prompt_tokens"] or 0) + (recs[u]["completion_tokens"] or 0) for u, _ in items])
    info = {"name": a.timing_run, "corpus": corpus, "split": split, "llm": a.llm, "n_calls": len(items),
            "wall_s": wall, "calls_per_s": len(items) / wall,
            "accelerator_s_per_1k_calls": 1000 * wall * lcfg.get("n_gpus", 1) / len(items),
            "latency_s": {"mean": float(lat.mean()), "median": float(np.median(lat)),
                          "p95": float(np.percentile(lat, 95))},
            "tokens_per_call_mean": float(tok.mean()), "environment": env,
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - wall))}
    if out.exists():                                          # keep answers, replace cost columns
        t = pd.read_pickle(out)
        key = t["uid"].map(lambda u: f"{corpus}|{u}")
        same = [parse_answer(recs[k]["raw"], [Candidate(c[0], c[1], c[2]) for c in cands])[0] == old
                for k, cands, old in zip(key, t["candidates"], t["llm_id"])]
        info["answers_identical_to_main_run"] = float(np.mean(same))
        t["llm_prompt_tokens"] = key.map(lambda u: recs[u]["prompt_tokens"])
        t["llm_completion_tokens"] = key.map(lambda u: recs[u]["completion_tokens"])
        t["llm_latency_s"] = key.map(lambda u: recs[u]["latency_s"])
        t["llm_gpu_s"] = key.map(lambda u: recs[u].get("gpu_s"))
        t.attrs["timing_run"] = a.timing_run
        t.to_pickle(out)
        print("updated cost columns in", out)
    else:
        print("note:", out, "not found; run without --timing-run first to create the answer table")
    path = tdir / f"{a.llm}__{corpus}_{split}__{a.timing_run}.json"
    path.write_text(json.dumps(info, indent=2))
    print(json.dumps({k: info[k] for k in ("n_calls", "wall_s", "calls_per_s", "accelerator_s_per_1k_calls",
                                          "answers_identical_to_main_run") if k in info}, indent=2, default=float))
    print("wrote", path)


if __name__ == "__main__":
    main()
