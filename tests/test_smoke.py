"""Smoke tests that run without GPUs, models or downloaded data.

pytest -q tests/
"""
import numpy as np
import pandas as pd

from gatelink import evaluate as ev
from gatelink import gate as G
from gatelink.data import normalise_ids, read_pubtator
from gatelink.features import ALL_FEATURES, extract
from gatelink.kb import KB
from gatelink.ner import _decode, Window, label_list
from gatelink.retrieve import find_abbreviations
from gatelink.schema import Candidate, Mention

PUBTATOR = """123|t|Famotidine-associated delirium.
123|a|A patient developed delirium after famotidine.
123\t0\t10\tFamotidine\tChemical\tD015738
123\t22\t30\tdelirium\tDisease\tD003693
123\t52\t60\tdelirium\tDisease\tMESH:D003693|D000001
123\tCID\tD015738\tD003693

"""


def test_pubtator(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text(PUBTATOR)
    docs = read_pubtator(p, type_map={"Chemical": "Chemical", "Disease": "Disease"})
    assert len(docs) == 1 and len(docs[0].mentions) == 3
    m = docs[0].mentions[2]
    assert docs[0].text[m.start:m.end] == "delirium" and m.gold_ids == ["D003693", "D000001"]
    assert normalise_ids("-1") == []


def test_abbreviations():
    pairs = find_abbreviations("Patients with chronic kidney disease (CKD) were enrolled.")
    assert pairs.get("CKD") == "chronic kidney disease"


def test_features_and_decode():
    kb = KB("toy")
    kb.add("D1", "delirium", preferred=True)
    kb.add("D2", "acute confusion", preferred=True)
    kb.parents["D2"].add("D1")
    ms = [Mention("d", 0, 8, "delirium", "Disease")]
    cands = [[Candidate("D1", "delirium", 1.0), Candidate("D2", "acute confusion", 0.7)]]
    f = extract(ms, cands, ["delirium"], kb_for=lambda t: kb)[0]
    assert set(ALL_FEATURES) <= set(f) and f["exact_top1"] == 1 and f["top12_related"] == 1
    labels = label_list(["Disease"])
    probs = np.full((4, 3), 0.05)
    probs[[0, 3], 0] = 0.9
    probs[1, 1] = probs[2, 2] = 0.9
    out = _decode(Window("d", 0, "a heart attack"), np.array([[0, 0], [2, 7], [8, 14], [0, 0]]), probs, labels)
    assert [(m.text, m.etype) for m in out] == [("heart attack", "Disease")]


def synthetic_table(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    margin = rng.random(n)
    ret_ok = rng.random(n) < 0.55 + 0.4 * margin                 # retriever is right when the margin is large
    llm_ok = rng.random(n) < 0.85
    rows = []
    for i in range(n):
        matched = rng.random() > 0.05
        gold = ["G%d" % i] if matched else []
        ret_id = gold[0] if (matched and ret_ok[i]) else "X%d" % i
        llm_id = (gold[0] if matched else "NIL") if llm_ok[i] else "Y%d" % i
        r = {"uid": str(i), "doc_id": str(i // 10), "etype": ["Disease", "Gene"][i % 2], "matched": matched,
             "gold_ids": gold, "ret_id": ret_id, "llm_id": llm_id, "llm_valid": True,
             "llm_prompt_tokens": 400, "llm_completion_tokens": 2, "llm_latency_s": 0.3, "llm_gpu_s": 0.05}
        for f in ALL_FEATURES:
            r[f] = rng.random()
        r["margin"], r["top1"] = margin[i], 0.5 + margin[i] / 2
        rows.append(r)
    return pd.DataFrame(rows)


def _write(tmp_path, name, df):
    df.to_pickle(tmp_path / f"{name}.pkl")
    gold = df[df["matched"]].groupby(["doc_id", "etype"]).size().unstack(fill_value=0)
    gold.to_csv(tmp_path / f"{name}_gold.csv")
    return tmp_path / f"{name}.pkl"


def test_gate_policies_and_experiment(tmp_path):
    from gatelink.experiment import run
    dev = _write(tmp_path, "toy_e2e_s0_dev__llm", synthetic_table(seed=1))
    test = _write(tmp_path, "toy_e2e_s0_test__llm", synthetic_table(seed=2))
    cfg = {"kind": "lr", "target": "error", "accuracy_delta": 0.5, "budget": 0.3, "epsilon": 0.02}
    summary = run(dev, test, cfg, tmp_path / "res", bootstrap=50)
    main = pd.read_csv(tmp_path / "res" / "main.csv").set_index("system")
    assert main.loc["S1 Retriever only", "share_llm"] == 0 and main.loc["S2 Always-on LLM", "share_llm"] == 100
    assert main.loc["S9 Oracle gate", "F1"] >= main.loc["S2 Always-on LLM", "F1"]
    assert 0 < main.loc["S6 GateLink-A", "share_llm"] < 100
    assert abs(main.loc["S7 GateLink-B", "share_llm"] - 30) < 6
    assert summary["gate_auroc_test"] > 0.6
    crc = pd.read_csv(tmp_path / "res" / "conformal.csv")
    assert (crc["share_llm"].diff().dropna() <= 1e-9).all()        # looser epsilon -> fewer LLM calls


def test_topk_inner_product():
    import pytest
    torch = pytest.importorskip("torch")
    from gatelink.retrieve import topk_inner_product
    emb = torch.nn.functional.normalize(torch.randn(1000, 32), dim=-1)
    q = emb[[3, 500]]
    s, i = topk_inner_product(torch, q, emb, 5, chunk=1)
    assert list(i[:, 0]) == [3, 500] and abs(s[0, 0] - 1) < 1e-5


def test_parse_answer():
    from gatelink.llm_rerank import parse_answer
    cs = [Candidate("A", "a", 0.9), Candidate("B", "b", 0.8)]
    assert parse_answer("2", cs) == ("B", True)
    assert parse_answer("0", cs) == ("NIL", True)
    assert parse_answer("The answer is 1.", cs) == ("A", True)
    assert parse_answer("WT1", cs)[1] is False          # a digit inside a word is not an answer
    assert parse_answer("7", cs)[1] is False            # out of range: falls back to the retriever


def test_species_tie_breaker():
    from gatelink.species import tie_breaker
    key = tie_breaker({"d1": "BRCA1 knockout mice and murine cells", "d2": "BRCA1 in patients"},
                      {"g_human": "9606", "g_mouse": "10090"})
    m1, m2 = Mention("d1", 0, 5, "BRCA1", "Gene"), Mention("d2", 0, 5, "BRCA1", "Gene")
    cands = [Candidate("g_human", "BRCA1", 1.0), Candidate("g_mouse", "Brca1", 1.0)]
    assert min(cands, key=lambda c: key(m1, c)).concept_id == "g_mouse"
    assert min(cands, key=lambda c: key(m2, c)).concept_id == "g_human"


def test_config():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs" / "gatelink.yaml").read_text())
    assert {"ncbi_disease", "bc5cdr", "nlm_chem", "nlm_gene_sp"} <= set(cfg["corpora"])
    assert [x["model"] for x in cfg["llm"]] == ["gemma4:12b", "gemma4:26b", "gemma4:31b"]
    assert cfg["gate"]["kind"] == "hgb" and cfg["retrieval"]["k"] == 20
