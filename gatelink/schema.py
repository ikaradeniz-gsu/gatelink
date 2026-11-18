"""Shared data structures used by every stage of the GateLink pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

NIL = "NIL"


@dataclass
class Mention:
    doc_id: str
    start: int                      # character offset in Document.text
    end: int
    text: str
    etype: str                      # Disease, Chemical, Gene, Species, CellLine, UMLS ...
    gold_ids: list[str] = field(default_factory=list)   # empty for predicted mentions
    # recognition signals (filled by the NER step; 1.0 / 0.0 for gold spans)
    span_prob: float = 1.0
    boundary_entropy: float = 0.0
    type_prob: float = 1.0

    @property
    def uid(self) -> str:
        return f"{self.doc_id}:{self.start}-{self.end}:{self.etype}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["uid"] = self.uid
        return d


@dataclass
class Document:
    doc_id: str
    text: str
    mentions: list[Mention] = field(default_factory=list)   # gold annotations


@dataclass
class Candidate:
    concept_id: str
    name: str                     # best-matching name of the concept
    score: float                  # retrieval score (cosine similarity)
    synonyms: Optional[list[str]] = None
