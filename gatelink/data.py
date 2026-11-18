"""Corpus loaders: PubTator (NCBI Disease, BC5CDR, BioRED, MedMentions) and BioC XML (NLM-Chem, NLM-Gene, GNormPlus).

Every loader returns list[Document] with gold Mention objects. Identifiers are normalised by
stripping database prefixes (MESH:, UMLS:, NCBIGene: ...) and splitting composite ids.
Mentions whose only identifier is "-1" or empty are dropped (no in-KB gold concept), as BELB does.
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Iterable

from .schema import Document, Mention

_PREFIX = re.compile(r"^(MESH|MeSH|UMLS|NCBIGene|NCBI Gene|NCBITaxon|NCBITaxonomy|CVCL|OMIM|CHEBI)[:_]", re.I)
_SPLIT = re.compile(r"[|,;]")


def normalise_ids(raw: str) -> list[str]:
    ids = []
    for part in _SPLIT.split(raw or ""):
        part = part.strip()
        if not part or part in {"-1", "-", "None"}:
            continue
        part = _PREFIX.sub("", part)
        ids.append(part)
    return ids


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")


# --------------------------------------------------------------------------- PubTator
def read_pubtator(path: str | Path, type_map: dict[str, str] | None = None,
                  default_type: str | None = None) -> list[Document]:
    """Parse a PubTator file. type_map maps raw types (e.g. 'DiseaseOrPhenotypicFeature', 'T047')
    to GateLink entity types; unmapped types are skipped unless default_type is given."""
    docs: list[Document] = []
    title, abstract, pmid, anns = "", "", None, []

    seen: set[str] = set()

    def flush():
        if pmid is None or pmid in seen:          # some corpora repeat a document (e.g. NCBI Disease 8528200)
            return
        seen.add(pmid)
        text = f"{title} {abstract}" if abstract else title
        doc = Document(pmid, text)
        for start, end, mtext, raw_type, raw_ids in anns:
            etype = (type_map or {}).get(raw_type, default_type)
            if etype is None:
                continue
            ids = normalise_ids(raw_ids)
            if not ids:
                continue
            doc.mentions.append(Mention(pmid, start, end, mtext, etype, ids))
        docs.append(doc)

    with _open(Path(path)) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                flush()
                title, abstract, pmid, anns = "", "", None, []
                continue
            if "|t|" in line[:40]:
                pmid, title = line.split("|t|", 1)
            elif "|a|" in line[:40]:
                pmid, abstract = line.split("|a|", 1)
            else:
                cols = line.split("\t")
                if len(cols) >= 6 and cols[1].isdigit():          # mention line
                    anns.append((int(cols[1]), int(cols[2]), cols[3], cols[4], cols[5]))
                # relation lines (BioRED) have a non-numeric 2nd column and are ignored
        flush()
    return docs


# --------------------------------------------------------------------------- BioC XML
def read_bioc(path: str | Path, type_map: dict[str, str] | None = None,
              id_keys: Iterable[str] = ("identifier", "Identifier", "NCBI Gene", "NCBI Gene identifier", "MESH"),
              default_type: str | None = None) -> list[Document]:
    """Parse a BioC XML collection (needs `pip install bioc`). Passage texts are placed at their
    offsets so that annotation offsets index Document.text directly."""
    from bioc import biocxml

    with _open(Path(path)) as fh:
        collection = biocxml.load(fh)
    docs = []
    for bdoc in collection.documents:
        end = max((p.offset + len(p.text or "") for p in bdoc.passages), default=0)
        chars = [" "] * end
        for p in bdoc.passages:
            for i, ch in enumerate(p.text or ""):
                chars[p.offset + i] = ch
        doc = Document(bdoc.id, "".join(chars))
        for p in bdoc.passages:
            for ann in p.annotations:
                raw_type = ann.infons.get("type", "")
                etype = (type_map or {}).get(raw_type, default_type)
                if etype is None or not ann.locations:
                    continue
                raw_ids = next((ann.infons[k] for k in id_keys if k in ann.infons), "")
                ids = normalise_ids(raw_ids)
                if not ids:
                    continue
                loc = ann.locations[0]                      # discontinuous mentions: first fragment
                doc.mentions.append(Mention(doc.doc_id, loc.offset, loc.offset + loc.length,
                                            ann.text or "", etype, ids))
        docs.append(doc)
    return docs


def load_corpus(spec: dict, split: str) -> list[Document]:
    """spec comes from configs/*.yaml: {format, files: {train, dev, test}, type_map, default_type}."""
    path = spec["files"][split]
    fmt = spec["format"]
    kwargs = dict(type_map=spec.get("type_map"), default_type=spec.get("default_type"))
    if fmt == "pubtator":
        docs = read_pubtator(path, **kwargs)
    elif fmt == "bioc":
        docs = read_bioc(path, id_keys=spec.get("id_keys", ("identifier", "Identifier")), **kwargs)
    else:
        raise ValueError(f"unknown format {fmt}")
    if spec.get("doc_ids", {}).get(split):          # corpora shipped as one file + id lists (e.g. MedMentions)
        keep = set(Path(spec["doc_ids"][split]).read_text().split())
        docs = [d for d in docs if d.doc_id in keep]
    return docs
