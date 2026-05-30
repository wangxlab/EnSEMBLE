"""Typed records used by the upstream clustering pipeline.

These are the data structures consumed by ``prefilter.py`` and ``themes.py``
(the locked clustering step). The agent's own input/output schemas (Pydantic
models) live in ``schemas.py``.

This module is intentionally minimal: only the classes needed by the
clustering step are kept. The v1.x ``ClaimEvidence``, ``HelperClaim``,
``EvidenceBundle``, and ``ESEARecord`` are obsolete in v2.0 (replaced by the
single-call classifier's ``VerdictItem`` in ``schemas.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class GSEARecord:
    term: str
    source: str | None
    nes: float
    q_value: float
    size: int
    direction: str
    score: float
    leading_edge: Tuple[str, ...]


@dataclass(frozen=True)
class HelperRecord:
    helper_name: str
    helper_class: str
    tf_family: str | None
    nes: float
    q_value: float
    direction: str
    size: int | None
    top_hallmark: str | None


@dataclass
class Theme:
    """Internal upstream Theme (a group of GSEA records). Distinct from
    the agent's Pydantic Theme schema in schemas.py."""

    label: str
    direction: str
    terms: List[GSEARecord]

    def top_terms(self, n: int) -> List[GSEARecord]:
        return sorted(self.terms, key=lambda rec: rec.score, reverse=True)[:n]


@dataclass
class ThemeSummary:
    theme_id: str
    label: str
    direction: str
    collection: str | None
    effect: float
    q_value: float
    top_pathways: List[GSEARecord]
    leading_edges: tuple[str, ...]
    helper_mean_effect: float | None = None
