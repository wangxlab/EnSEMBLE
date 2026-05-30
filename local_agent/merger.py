"""Post-clustering theme merger (v2.2.1).

Runs after the locked clustering pipeline. Merges any pair of same-direction
themes whose top leading-edge gene sets overlap by >= a Jaccard threshold
(default 0.5). Uses union-find for transitive merges.

Why this exists:
  Datasets with many significant pathways (e.g. iPSC, N=775) produce
  near-redundant clusters even at the coarsest dynamicTreeCut setting
  (deepSplit=1, minClusterSize=10). For example, iPSC DOWN has 5 separate
  cell-cycle clusters that share MAD2L1/AURKA/INCENP/PLK1/CCNB1 in their
  leading edges -- biologically one finding, statistically five clusters.

What it does NOT do:
  - Touch the upstream clustering pipeline (locked).
  - Cross directions (UP themes only merge with UP, DOWN with DOWN).
  - Merge below the configured Jaccard threshold.

Default threshold = 0.5 (set by AgentConfig.merge_jaccard_threshold).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .schemas import Theme


# ---------- Jaccard helpers --------------------------------------------------


def jaccard(a: List[str], b: List[str]) -> float:
    """Symmetric Jaccard on two ordered lists treated as sets."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------- Union-find for transitive grouping -------------------------------


class _DSU:
    def __init__(self, items: List[str]) -> None:
        self._parent: Dict[str, str] = {x: x for x in items}

    def find(self, x: str) -> str:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self._parent[ry] = rx

    def groups(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for x in self._parent:
            out.setdefault(self.find(x), []).append(x)
        return out


# ---------- Merge log structures ---------------------------------------------


@dataclass
class MergedThemeRecord:
    """One non-trivial merge group (size > 1)."""

    direction: str
    representative_id: str
    representative_label: str
    representative_nes: float
    members: List[Dict] = field(default_factory=list)  # {theme_id, label, mean_nes, jaccard_with_rep}
    shared_genes: List[str] = field(default_factory=list)


@dataclass
class MergeLog:
    dataset_id: str
    jaccard_threshold: float
    n_before_total: int
    n_before_up: int
    n_before_down: int
    n_after_total: int
    n_after_up: int
    n_after_down: int
    groups: List[MergedThemeRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "jaccard_threshold": self.jaccard_threshold,
            "before": {
                "total": self.n_before_total,
                "UP": self.n_before_up,
                "DOWN": self.n_before_down,
            },
            "after": {
                "total": self.n_after_total,
                "UP": self.n_after_up,
                "DOWN": self.n_after_down,
            },
            "n_groups_merged": len(self.groups),
            "n_themes_folded": sum(len(g.members) for g in self.groups),
            "groups": [
                {
                    "direction": g.direction,
                    "representative_id": g.representative_id,
                    "representative_label": g.representative_label,
                    "representative_nes": g.representative_nes,
                    "merged_with": g.members,
                    "shared_genes": g.shared_genes,
                }
                for g in self.groups
            ],
        }


# ---------- Helpers ----------------------------------------------------------


_THEME_ID_PREFIX = re.compile(r"^(up|down)_(c\d+|orphan)_")


def _normalize_theme_id_for_merge(rep_id: str, members_count: int) -> str:
    """If the representative is a 'cN' or orphan id, optionally tag the
    merged theme so it's clear it absorbed others. We keep it simple and
    just append '_merged{n}' when n > 1.

    Example: down_c1_e2f-targets + 4 others -> down_c1_e2f-targets_merged5
    """
    if members_count <= 1:
        return rep_id
    return f"{rep_id}_merged{members_count}"


def _ordered_union_genes(themes_in_group: List[Theme], cap: int = 24) -> List[str]:
    """Union the top_leading_edge_genes across a group, ordered by frequency
    (most-shared first), then by appearance order in the representative.

    Cap at `cap` to keep payload manageable. Default 24 = 2x typical (12).
    """
    counts: Dict[str, int] = {}
    first_pos: Dict[str, int] = {}
    for ti, theme in enumerate(themes_in_group):
        for pi, gene in enumerate(theme.top_leading_edge_genes or []):
            counts[gene] = counts.get(gene, 0) + 1
            first_pos.setdefault(gene, ti * 100 + pi)
    ordered = sorted(
        counts.keys(), key=lambda g: (-counts[g], first_pos[g])
    )
    return ordered[:cap]


def _ordered_union_pathways(themes_in_group: List[Theme]) -> List[str]:
    """Union pathway names, preserving rep's order, then add new from others."""
    seen: set = set()
    out: List[str] = []
    for theme in themes_in_group:
        for p in theme.top_pathways or []:
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def _shared_genes(themes_in_group: List[Theme]) -> List[str]:
    """Genes appearing in >= half the group (by count). Reported in the log."""
    if len(themes_in_group) <= 1:
        return []
    counts: Dict[str, int] = {}
    for theme in themes_in_group:
        for g in set(theme.top_leading_edge_genes or []):
            counts[g] = counts.get(g, 0) + 1
    threshold = max(2, (len(themes_in_group) + 1) // 2)
    return sorted([g for g, c in counts.items() if c >= threshold], key=lambda g: -counts[g])


# ---------- Main merge function ---------------------------------------------


def merge_redundant_themes(
    themes: List[Theme],
    jaccard_threshold: float = 0.5,
    dataset_id: str = "?",
    full_le_lookup: Optional[Dict[str, set]] = None,
) -> Tuple[List[Theme], MergeLog]:
    """Merge same-direction themes whose leading-edge gene sets overlap >= threshold.

    If `full_le_lookup` is provided, it is used as {theme_id: set(genes)} for the
    Jaccard calculation. Otherwise the merger falls back to each theme's
    top_leading_edge_genes (typically capped at 12 genes, which under-counts
    real biological overlap).

    Returns (merged_themes, log).
    """
    log = MergeLog(
        dataset_id=dataset_id,
        jaccard_threshold=jaccard_threshold,
        n_before_total=len(themes),
        n_before_up=sum(1 for t in themes if t.direction == "UP"),
        n_before_down=sum(1 for t in themes if t.direction == "DOWN"),
        n_after_total=0,
        n_after_up=0,
        n_after_down=0,
    )

    if jaccard_threshold <= 0 or jaccard_threshold > 1:
        # No-op: jaccard <= 0 disables, > 1 impossible
        log.n_after_total = len(themes)
        log.n_after_up = log.n_before_up
        log.n_after_down = log.n_before_down
        return list(themes), log

    by_dir: Dict[str, List[Theme]] = {"UP": [], "DOWN": []}
    for t in themes:
        by_dir.setdefault(t.direction, []).append(t)

    def _le_set(t: Theme) -> set:
        if full_le_lookup is not None and t.theme_id in full_le_lookup:
            s = full_le_lookup[t.theme_id]
            if s:
                return s
        return set(t.top_leading_edge_genes or [])

    merged_all: List[Theme] = []

    for direction in ("UP", "DOWN"):
        bucket = by_dir.get(direction, [])
        if not bucket:
            continue
        ids = [t.theme_id for t in bucket]
        by_id: Dict[str, Theme] = {t.theme_id: t for t in bucket}
        dsu = _DSU(ids)

        # Pre-compute LE sets once.
        le_sets: Dict[str, set] = {t.theme_id: _le_set(t) for t in bucket}

        # Build edges between ids whose Jaccard meets threshold.
        edge_jaccards: Dict[Tuple[str, str], float] = {}
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                ti, tj = bucket[i], bucket[j]
                a, b = le_sets[ti.theme_id], le_sets[tj.theme_id]
                if not a and not b:
                    jac = 0.0
                else:
                    jac = len(a & b) / len(a | b)
                if jac >= jaccard_threshold:
                    dsu.union(ti.theme_id, tj.theme_id)
                    key = tuple(sorted([ti.theme_id, tj.theme_id]))
                    edge_jaccards[key] = jac

        for root, members_ids in dsu.groups().items():
            members = [by_id[i] for i in members_ids]
            # Representative selection (priority order):
            #   1. HALLMARK_* present in top_pathways[0] (most specific signal)
            #   2. Any HALLMARK_* in top_pathways
            #   3. Strongest |mean_nes|
            #   4. Alphabetical theme_id (deterministic tiebreak)
            # This prevents generic GO themes (Protein Modification Process,
            # Cellular Homeostasis, etc.) from absorbing specific HALLMARK-
            # anchored themes (EMT, MYC targets, IFN response).
            def _rep_score(t: Theme) -> tuple:
                pathways = [p.upper() for p in (t.top_pathways or [])]
                hallmark_in_pos0 = bool(pathways and pathways[0].startswith("HALLMARK_"))
                any_hallmark = any(p.startswith("HALLMARK_") for p in pathways)
                # Lower score = picked first; negate for "higher is better" fields.
                return (
                    0 if hallmark_in_pos0 else 1,
                    0 if any_hallmark else 1,
                    -abs(t.mean_nes),
                    t.theme_id,
                )
            members.sort(key=_rep_score)
            rep = members[0]

            # Build the merged theme.
            if len(members) == 1:
                merged_all.append(rep)
                continue

            new_label = rep.label
            new_pathways = _ordered_union_pathways(members)
            new_genes = _ordered_union_genes(members, cap=24)
            new_n_members = sum(t.n_members for t in members)
            new_id = _normalize_theme_id_for_merge(rep.theme_id, len(members))

            merged_theme = Theme(
                theme_id=new_id,
                label=new_label,
                direction=direction,
                mean_nes=rep.mean_nes,
                n_members=new_n_members,
                top_pathways=new_pathways,
                top_leading_edge_genes=new_genes,
            )
            merged_all.append(merged_theme)

            # Log
            shared = _shared_genes(members)
            record = MergedThemeRecord(
                direction=direction,
                representative_id=rep.theme_id,
                representative_label=rep.label,
                representative_nes=rep.mean_nes,
                shared_genes=shared,
            )
            for m in members[1:]:
                key = tuple(sorted([rep.theme_id, m.theme_id]))
                jac = edge_jaccards.get(key)
                # If no direct edge, compute (transitive merge case).
                if jac is None:
                    a, b = le_sets[rep.theme_id], le_sets[m.theme_id]
                    if not a and not b:
                        jac = 0.0
                    else:
                        jac = len(a & b) / len(a | b)
                record.members.append(
                    {
                        "theme_id": m.theme_id,
                        "label": m.label,
                        "mean_nes": m.mean_nes,
                        "jaccard_with_rep": round(jac, 3),
                    }
                )
            log.groups.append(record)

    # Stable order: highest |mean_nes| first within direction, UP before DOWN.
    merged_all.sort(key=lambda t: (0 if t.direction == "UP" else 1, -abs(t.mean_nes)))

    log.n_after_total = len(merged_all)
    log.n_after_up = sum(1 for t in merged_all if t.direction == "UP")
    log.n_after_down = sum(1 for t in merged_all if t.direction == "DOWN")

    return merged_all, log


# ---------- Markdown report -------------------------------------------------


def render_merge_log_markdown(log: MergeLog) -> str:
    lines: List[str] = []
    lines.append(f"# Post-clustering merge log — {log.dataset_id}")
    lines.append("")
    lines.append(f"- Jaccard threshold: **{log.jaccard_threshold}**")
    lines.append(f"- Before: **{log.n_before_total}** themes ({log.n_before_up} UP, {log.n_before_down} DOWN)")
    lines.append(f"- After: **{log.n_after_total}** themes ({log.n_after_up} UP, {log.n_after_down} DOWN)")
    n_folded = sum(len(g.members) for g in log.groups)
    lines.append(f"- {len(log.groups)} merge group(s); {n_folded} theme(s) folded")
    lines.append("")
    if not log.groups:
        lines.append("_No themes merged at this threshold._")
        return "\n".join(lines) + "\n"

    for direction in ("UP", "DOWN"):
        groups = [g for g in log.groups if g.direction == direction]
        if not groups:
            continue
        lines.append(f"## {direction} merge groups")
        lines.append("")
        for g in sorted(groups, key=lambda g: -abs(g.representative_nes)):
            lines.append(f"### {g.representative_label}  ({g.representative_id}, NES={g.representative_nes:+.2f})")
            if g.shared_genes:
                shared = ", ".join(g.shared_genes[:8])
                lines.append(f"_Shared genes (in >= half the group): {shared}_")
            lines.append("")
            lines.append("| theme_id | label | NES | Jaccard w/ rep |")
            lines.append("|---|---|---|---|")
            for m in sorted(g.members, key=lambda m: -m["jaccard_with_rep"]):
                lines.append(
                    f"| {m['theme_id']} | {m['label'][:40]} | {m['mean_nes']:+.2f} | {m['jaccard_with_rep']:.2f} |"
                )
            lines.append("")
    return "\n".join(lines) + "\n"
