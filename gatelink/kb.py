"""Knowledge-base loaders. Each returns a KB: concept ids, names (synonyms), parents and extra info.

Supported sources (download links in README):
  ctd        CTD_diseases.tsv.gz / CTD_chemicals.tsv.gz   (MeSH/OMIM ids, parents, synonyms)
  gene_info  NCBI gene_info.gz, filtered to a list of taxa (all-species gene_info is very large)
  taxonomy   NCBI Taxonomy names.dmp (from taxdump.tar.gz)
  umls       MRCONSO.RRF (+ MRSTY.RRF to keep only selected semantic types, e.g. MedMentions ST21pv)
  tsv        generic file: id<TAB>name[<TAB>parent1|parent2]  (e.g. exported from BELB or Cellosaurus)
"""
from __future__ import annotations

import gzip
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[\-‐‑–—_/,.;:()\[\]{}'\"]")


def norm_name(s: str) -> str:
    """Normalisation used for exact/dictionary matching."""
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


@dataclass
class KB:
    name: str
    preferred: dict[str, str] = field(default_factory=dict)            # cid -> preferred name
    synonyms: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    parents: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    info: dict[str, str] = field(default_factory=dict)                  # cid -> short context (species, definition)
    alt_to_primary: dict[str, str] = field(default_factory=dict)        # merged/alternative id -> KB concept id
    taxon: dict[str, str] = field(default_factory=dict)                 # cid -> NCBI taxon (genes only)

    def add(self, cid: str, name: str, preferred: bool = False):
        if not name:
            return
        if preferred or cid not in self.preferred:
            self.preferred[cid] = name
        if name not in self.synonyms[cid]:
            self.synonyms[cid].append(name)

    # ---- derived lookups (built lazily)
    def name_table(self) -> list[tuple[str, str]]:
        """All (name, cid) pairs, one row per synonym."""
        return [(n, cid) for cid, names in self.synonyms.items() for n in names]

    def exact_index(self) -> dict[str, set[str]]:
        if not hasattr(self, "_exact"):
            idx: dict[str, set[str]] = defaultdict(set)
            for n, cid in self.name_table():
                idx[norm_name(n)].add(cid)
            self._exact = idx
        return self._exact

    def related(self, a: str, b: str) -> bool:
        """True if a and b are parent/child or siblings."""
        pa, pb = self.parents.get(a, set()), self.parents.get(b, set())
        return b in pa or a in pb or bool(pa & pb)


def _open(path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if str(path).endswith(".gz") \
        else open(path, encoding="utf-8", errors="replace")


_IDTOK = re.compile(r"^[A-Za-z]+:[A-Za-z0-9:._-]+$")


def _is_id_list(field: str) -> bool:
    toks = [t for t in field.split("|") if t]
    return bool(toks) and all(_IDTOK.match(t) for t in toks)


def _strip(cid: str) -> str:
    return re.sub(r"^(MESH|OMIM):", "", cid)


_CTD_FIELDS = {"ChemicalName", "ChemicalID", "DiseaseName", "DiseaseID", "Synonyms", "ParentIDs", "Definition"}


def _ctd_header(path) -> list[str] | None:
    """CTD files start with comment lines; one of them lists the column names ("# ChemicalName<TAB>ChemicalID...").
    Columns are read by name when that line is present, because CTD adds columns between releases
    (e.g. PubChem CID and DTXSID in the chemicals file), which shifts every later position."""
    with _open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                return None
            cols = [c.strip() for c in line.lstrip("#").rstrip("\n").split("\t")]
            if len(cols) > 3 and len(_CTD_FIELDS & set(cols)) >= 3:
                return cols
    return None


def load_ctd(path) -> KB:
    kb = KB(Path(path).name)
    header = _ctd_header(path)
    if header:
        col = {h: i for i, h in enumerate(header)}
        i_name = col.get("ChemicalName", col.get("DiseaseName", 0))
        i_id = col.get("ChemicalID", col.get("DiseaseID", 1))
        i_syn = [col[h] for h in header if "Synonyms" in h]          # Synonyms, CTD Curated Synonyms, ...
        i_par, i_def = col.get("ParentIDs"), col.get("Definition")
        i_alt = col.get("AltDiseaseIDs")
    with _open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if header:
                if len(c) <= max(i_name, i_id):
                    continue
                get = lambda i: c[i] if i is not None and i < len(c) else ""
                name, cid = c[i_name], _strip(c[i_id])
                syns = [x for i in i_syn for x in get(i).split("|")]
                parents, def_col, alt_col = get(i_par), get(i_def), get(i_alt)
            else:
                # No header line: fall back to the documented positions, detecting the id-list column by content.
                # diseases  (9 cols): Name, ID, {AltDiseaseIDs, Definition} in either order, ParentIDs, TreeNumbers,
                #                     ParentTreeNumbers, Synonyms, SlimMappings
                # chemicals (8-9 cols): Name, ID, CasRN, Definition, ParentIDs, TreeNumbers, ParentTreeNumbers,
                #                     Synonyms, [DrugBankIDs]
                if len(c) < 8:
                    continue
                name, cid = c[0], _strip(c[1])
                if _is_id_list(c[3]) and not _is_id_list(c[2]):
                    alt_col, def_col = c[3], c[2]
                elif _is_id_list(c[2]):
                    alt_col, def_col = c[2], c[3]
                else:                                               # chemicals: c[2] is a CAS number
                    alt_col, def_col = "", c[3]
                syns, parents = c[7].split("|"), c[4]
            kb.add(cid, name, preferred=True)
            for syn in filter(None, syns):
                kb.add(cid, syn)
            kb.parents[cid].update(_strip(p) for p in filter(None, parents.split("|")))
            if def_col and not _is_id_list(def_col):
                kb.info[cid] = def_col[:200]
            for alt in filter(None, alt_col.split("|")):       # OMIM/MeSH ids merged into this concept (MEDIC)
                if alt.startswith(("MESH:", "OMIM:")):
                    kb.alt_to_primary.setdefault(_strip(alt), cid)
    return kb


def load_gene_info(path, taxa: set[str] | None = None, taxon_names: dict[str, str] | None = None) -> KB:
    kb = KB("ncbi_gene")
    with _open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            tax, gid = c[0], c[1]
            if taxa and tax not in taxa:
                continue
            kb.add(gid, c[2], preferred=True)                          # symbol
            for syn in c[4].split("|"):
                if syn != "-":
                    kb.add(gid, syn)
            if len(c) > 11 and c[11] != "-":
                kb.add(gid, c[11])                                     # full name (authority)
            elif c[8] != "-":
                kb.add(gid, c[8])                                      # description
            for od in (c[13].split("|") if len(c) > 13 and c[13] != "-" else [])[:5]:
                kb.add(gid, od)
            species = (taxon_names or {}).get(tax, f"taxon {tax}")
            kb.info[gid] = f"{species}; {c[8]}"
            kb.taxon[gid] = tax
    return kb


def load_taxonomy(names_dmp, classes=("scientific name", "synonym", "genbank common name",
                                      "common name", "equivalent name", "blast name"),
                  taxa: set[str] | None = None) -> KB:
    kb = KB("ncbi_taxonomy")
    with _open(names_dmp) as fh:
        for line in fh:
            c = [x.strip() for x in line.split("|")]
            tax, name, cls = c[0], c[1], c[3]
            if cls not in classes or (taxa and tax not in taxa):
                continue
            kb.add(tax, name, preferred=(cls == "scientific name"))
    return kb


def load_umls(mrconso, mrsty=None, tuis: set[str] | None = None, sabs: set[str] | None = None) -> KB:
    keep = None
    if mrsty and tuis:
        keep = set()
        with _open(mrsty) as fh:
            for line in fh:
                c = line.split("|")
                if c[1] in tuis:
                    keep.add(c[0])
    kb = KB("umls")
    with _open(mrconso) as fh:
        for line in fh:
            c = line.split("|")
            cui, lat, ispref, sab, string, suppress = c[0], c[1], c[6], c[11], c[14], c[16]
            if lat != "ENG" or suppress in {"O", "E"} or (keep is not None and cui not in keep):
                continue
            if sabs and sab not in sabs:
                continue
            kb.add(cui, string, preferred=(ispref == "Y" and c[2] == "P"))
    return kb


def load_cellosaurus(path) -> KB:
    """cellosaurus.txt flat file: records separated by '//', fields ID (name), AC (CVCL_ accession),
    SY (synonyms, '; '-separated), OX (species), HI (parent cell line)."""
    kb = KB("cellosaurus")
    rec: dict[str, list[str]] = defaultdict(list)

    def flush():
        if rec.get("AC"):
            cid = rec["AC"][0].replace("CVCL_", "")
            kb.add(cid, rec["ID"][0] if rec.get("ID") else cid, preferred=True)
            for line in rec.get("SY", []):
                for s in line.split("; "):
                    kb.add(cid, s.strip())
            for hi in rec.get("HI", []):
                kb.parents[cid].add(hi.split()[0].replace("CVCL_", ""))
            if rec.get("OX"):
                kb.info[cid] = rec["OX"][0].split("! ")[-1]

    with _open(path) as fh:
        for line in fh:
            if line.startswith("//"):
                flush()
                rec = defaultdict(list)
            elif len(line) > 5 and line[2:5] == "   ":
                rec[line[:2]].append(line[5:].rstrip("\n"))
        flush()
    return kb


def load_tsv(path, name="tsv") -> KB:
    kb = KB(name)
    with _open(path) as fh:
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) < 2 or line.startswith("#"):
                continue
            kb.add(c[0], c[1])
            if len(c) > 2 and c[2]:
                kb.parents[c[0]].update(c[2].split("|"))
    return kb


def load_kb(spec: dict) -> KB:
    kind = spec["kind"]
    if kind == "ctd":
        return load_ctd(spec["path"])
    if kind == "gene_info":
        return load_gene_info(spec["path"], set(map(str, spec.get("taxa", []))) or None, spec.get("taxon_names"))
    if kind == "taxonomy":
        return load_taxonomy(spec["path"])
    if kind == "umls":
        return load_umls(spec["mrconso"], spec.get("mrsty"), set(spec.get("tuis", [])) or None,
                         set(spec.get("sabs", [])) or None)
    if kind == "cellosaurus":
        return load_cellosaurus(spec["path"])
    if kind == "tsv":
        return load_tsv(spec["path"], spec.get("name", "tsv"))
    raise ValueError(kind)
