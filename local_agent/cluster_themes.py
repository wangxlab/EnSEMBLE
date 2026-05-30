"""Leading-edge Jaccard -> hierarchical -> dynamicTreeCut -> representative label.

Per-direction theme generator that replaces the v6 regex/YAML taxonomy. It
operates on :class:`GSEARecord` objects, so it slots into the existing v6
pipeline without touching downstream LLM stages: :func:`cluster_records` is
consumed by :func:`themes.build_theme_summaries`, which returns the same
``Dict[str, List[ThemeSummary]]`` shape the rest of the pipeline expects.

Deterministic given the same input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from dynamicTreeCut import cutreeHybrid
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from .data_models import GSEARecord


@dataclass
class ClusteredPathway:
    term: str
    nes: float
    q_value: float
    size: int
    leading_edge_n: int
    cluster: int                   # 0 = orphan
    centrality: float              # mean Jaccard to other cluster members
    is_representative: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClusteredTheme:
    cluster: int                   # 1-based; 0 reserved for orphans
    label: str                     # representative term (cleaned)
    representative_term: str
    n_members: int
    mean_nes: float
    min_q_value: float
    mean_intra_similarity: float
    members: List[ClusteredPathway] = field(default_factory=list)
    # indices back into the original record list (for plot bookkeeping)
    member_indices: Tuple[int, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        d = {
            "cluster": self.cluster,
            "label": self.label,
            "representative_term": self.representative_term,
            "n_members": self.n_members,
            "mean_nes": self.mean_nes,
            "min_q_value": self.min_q_value,
            "mean_intra_similarity": self.mean_intra_similarity,
            "members": [m.to_dict() for m in self.members],
        }
        return d


@dataclass
class ClusteringResult:
    themes: List[ClusteredTheme]
    orphans: List[ClusteredPathway]
    params: dict
    # raw arrays kept for plotting / auditing (not serialized by default)
    similarity_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    records: List[GSEARecord] = field(default_factory=list)
    cluster_labels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    def to_json_dict(self) -> dict:
        return {
            "params": self.params,
            "n_input_pathways": len(self.records),
            "n_themes": len(self.themes),
            "n_orphans": len(self.orphans),
            "themes": [t.to_dict() for t in self.themes],
            "orphans": [o.to_dict() for o in self.orphans],
        }


# -- helpers -----------------------------------------------------------------


def _jaccard_matrix(gene_sets: Sequence[frozenset]) -> np.ndarray:
    n = len(gene_sets)
    sim = np.ones((n, n), dtype=float)
    for i in range(n):
        a = gene_sets[i]
        if not a:
            for j in range(n):
                if i != j:
                    sim[i, j] = sim[j, i] = 0.0
            continue
        for j in range(i + 1, n):
            b = gene_sets[j]
            if not b:
                sim[i, j] = sim[j, i] = 0.0
                continue
            inter = len(a & b)
            if inter == 0:
                sim[i, j] = sim[j, i] = 0.0
                continue
            union = len(a | b)
            sim[i, j] = sim[j, i] = inter / union
    return sim


_PREFIX_RE = re.compile(
    r"^(REACTOME_|GOBP_|GOCC_|GOMF_|HALLMARK_|KEGG_|WP_|PID_|BIOCARTA_)"
)


# -- name-anchor labeling -----------------------------------------------------
# Small curated list of canonical biology tokens. If any cluster member's term
# matches one of these anchors, the anchor's pretty label is promoted over the
# medoid-derived label. This keeps named minority signals (e.g., a single
# HALLMARK_EMT pathway in an otherwise metabolic cluster) from being buried
# under a metabolic medoid.
#
# Rules for adding entries: only well-known, unambiguous process names. Patterns
# are matched against the ORIGINAL (uppercase, underscored) term string after
# stripping MSigDB prefixes. First pattern to match a member wins within a
# cluster; if multiple members match different anchors, the member with the
# strongest score (|NES|) picks the anchor.
#
# Order matters: put narrower patterns before broader ones so e.g.
# INTERFERON_GAMMA matches before a generic INTERFERON.
_NAME_ANCHORS: Tuple[Tuple[str, str], ...] = (
    # Oncogenic / proliferative programs
    (r"HALLMARK_MYC_TARGETS(_V[12])?|MYC_TARGET", "MYC targets"),
    (r"HALLMARK_E2F_TARGETS|E2F_TARGET", "E2F targets"),
    (r"HALLMARK_G2M_CHECKPOINT", "G2/M checkpoint"),
    (r"HALLMARK_MITOTIC_SPINDLE", "Mitotic spindle"),
    (r"HALLMARK_DNA_REPAIR", "DNA repair"),
    # EMT / TGF-β / Wnt
    (r"HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION|EPITHELIAL_MESENCHYMAL_TRANSITION", "Epithelial-mesenchymal transition"),
    (r"HALLMARK_TGF_BETA_SIGNALING|TGFB1?_SIGNALING|TGF_BETA_SIGNALING", "TGF-β signaling"),
    (r"HALLMARK_WNT_BETA_CATENIN_SIGNALING|BETA_CATENIN|WNT_BETA_CATENIN|WNT_SIGNALING", "Wnt/β-catenin signaling"),
    (r"HALLMARK_NOTCH_SIGNALING|NOTCH_SIGNALING", "Notch signaling"),
    (r"HALLMARK_HEDGEHOG_SIGNALING", "Hedgehog signaling"),
    # Immune / cytokine / IFN
    (r"HALLMARK_TNFA_SIGNALING_VIA_NFKB|TNF_ALPHA_SIGNALING_VIA_NFKB", "TNFα/NF-κB signaling"),
    (r"HALLMARK_INFLAMMATORY_RESPONSE|INFLAMMATORY_RESPONSE", "Inflammatory response"),
    (r"HALLMARK_INTERFERON_GAMMA_RESPONSE|INTERFERON_GAMMA_RESPONSE|TYPE_II_INTERFERON", "IFN-γ response"),
    (r"HALLMARK_INTERFERON_ALPHA_RESPONSE|INTERFERON_ALPHA_RESPONSE|TYPE_I_INTERFERON", "IFN-α response"),
    (r"HALLMARK_IL6_JAK_STAT3_SIGNALING|JAK_STAT", "IL6/JAK-STAT signaling"),
    (r"HALLMARK_IL2_STAT5_SIGNALING", "IL2-STAT5 signaling"),
    (r"HALLMARK_COMPLEMENT", "Complement"),
    (r"HALLMARK_ALLOGRAFT_REJECTION", "Allograft rejection / immune"),
    # Stress / metabolism
    (r"HALLMARK_HYPOXIA|HYPOXIA_RESPONSE", "Hypoxia"),
    (r"HALLMARK_OXIDATIVE_PHOSPHORYLATION|OXIDATIVE_PHOSPHORYLATION", "Oxidative phosphorylation"),
    (r"HALLMARK_GLYCOLYSIS", "Glycolysis"),
    (r"HALLMARK_UNFOLDED_PROTEIN_RESPONSE", "Unfolded protein response"),
    (r"HALLMARK_P53_PATHWAY", "p53 pathway"),
    (r"HALLMARK_APOPTOSIS|INTRINSIC_APOPTOTIC", "Apoptosis"),
    # Growth / survival signaling
    (r"HALLMARK_MTORC1_SIGNALING|MTORC1|MTOR_SIGNALING", "mTORC1 signaling"),
    (r"HALLMARK_PI3K_AKT_MTOR_SIGNALING|PI3K_AKT", "PI3K/AKT/mTOR"),
    (r"HALLMARK_KRAS_SIGNALING_UP|KRAS_SIGNALING", "KRAS signaling"),
    (r"HALLMARK_ANDROGEN_RESPONSE", "Androgen response"),
    (r"HALLMARK_ESTROGEN_RESPONSE_(EARLY|LATE)", "Estrogen response"),
    (r"HALLMARK_ANGIOGENESIS", "Angiogenesis"),
)

_COMPILED_ANCHORS = tuple((re.compile(pat), label) for pat, label in _NAME_ANCHORS)


def _anchor_label(member_terms_and_scores: Sequence[Tuple[str, float]]) -> Tuple[str | None, str | None]:
    """Return (matched_term, anchor_label) for the strongest-|NES| member whose
    term matches any anchor regex. Returns (None, None) if no member matches.
    """
    best: Tuple[float, str, str] | None = None  # (score, matched_term, anchor_label)
    for term, score in member_terms_and_scores:
        stripped = _PREFIX_RE.sub("", term)
        full = term  # preserve prefix so HALLMARK_* anchors can match
        for pat, label in _COMPILED_ANCHORS:
            if pat.search(full) or pat.search(stripped):
                if best is None or abs(score) > best[0]:
                    best = (abs(score), term, label)
                break
    if best is None:
        return (None, None)
    return (best[1], best[2])


def clean_label(term: str, maxlen: int = 48) -> str:
    cleaned = _PREFIX_RE.sub("", term)
    cleaned = cleaned.replace("_", " ").strip()
    tokens = cleaned.split()
    pretty = []
    for tok in tokens:
        if tok.isupper() and len(tok) <= 4:
            pretty.append(tok)
        else:
            pretty.append(tok.title())
    label = " ".join(pretty)
    if len(label) > maxlen:
        label = label[: maxlen - 1].rstrip() + "\u2026"
    return label


# -- main entry --------------------------------------------------------------


def cluster_records(
    records: Sequence[GSEARecord],
    *,
    linkage_method: str = "average",
    deep_split: int = 2,
    min_cluster_size: int = 3,
    min_similarity: float = 0.0,
    pam_stage: bool = True,
) -> ClusteringResult:
    """Cluster GSEA records by leading-edge Jaccard and assign representative labels."""
    recs = [r for r in records if r.leading_edge]
    n = len(recs)
    params = {
        "linkage_method": linkage_method,
        "deep_split": deep_split,
        "min_cluster_size": min_cluster_size,
        "min_similarity": min_similarity,
        "pam_stage": pam_stage,
    }
    if n == 0:
        return ClusteringResult(themes=[], orphans=[], params=params)

    gene_sets = [frozenset(r.leading_edge) for r in recs]
    sim = _jaccard_matrix(gene_sets)
    if min_similarity > 0:
        sim = np.where(sim < min_similarity, 0.0, sim)
        np.fill_diagonal(sim, 1.0)

    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 1.0)
    dist = (dist + dist.T) / 2.0

    if n == 1:
        only_rec = recs[0]
        only = ClusteredPathway(
            term=only_rec.term, nes=only_rec.nes, q_value=only_rec.q_value,
            size=only_rec.size, leading_edge_n=len(gene_sets[0]),
            cluster=0, centrality=0.0, is_representative=False,
        )
        return ClusteringResult(
            themes=[], orphans=[only], params=params,
            similarity_matrix=sim, records=list(recs),
            cluster_labels=np.zeros(1, dtype=int),
        )

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=linkage_method)

    cut = cutreeHybrid(
        link=Z, distM=dist,
        deepSplit=deep_split, minClusterSize=min_cluster_size,
        pamStage=pam_stage, pamRespectsDendro=True,
        respectSmallClusters=True, verbose=0,
    )
    labels = np.asarray(cut["labels"], dtype=int)

    unique_clusters = sorted({int(c) for c in labels if int(c) != 0})
    themes: List[ClusteredTheme] = []
    orphans: List[ClusteredPathway] = []

    for ci in unique_clusters:
        member_idx = [i for i, c in enumerate(labels) if int(c) == ci]
        if len(member_idx) > 1:
            sub = sim[np.ix_(member_idx, member_idx)].copy()
            np.fill_diagonal(sub, np.nan)
            centralities = np.nanmean(sub, axis=1)
        else:
            centralities = np.zeros(len(member_idx))

        order = sorted(
            range(len(member_idx)),
            key=lambda k: (
                -centralities[k],
                recs[member_idx[k]].q_value,
                recs[member_idx[k]].term,
            ),
        )
        rep_local = order[0]
        rep_rec = recs[member_idx[rep_local]]

        members: List[ClusteredPathway] = []
        for k, idx in enumerate(member_idx):
            members.append(
                ClusteredPathway(
                    term=recs[idx].term,
                    nes=recs[idx].nes,
                    q_value=recs[idx].q_value,
                    size=recs[idx].size,
                    leading_edge_n=len(gene_sets[idx]),
                    cluster=ci,
                    centrality=float(centralities[k]),
                    is_representative=(k == rep_local),
                )
            )
        members.sort(key=lambda m: (-m.centrality, m.q_value))

        if len(member_idx) > 1:
            sub = sim[np.ix_(member_idx, member_idx)].copy()
            iu = np.triu_indices(len(member_idx), k=1)
            mean_intra = float(np.mean(sub[iu])) if iu[0].size else 0.0
        else:
            mean_intra = 0.0

        # Option-2 label override: if any cluster member matches a curated
        # name-anchor (HALLMARK_EMT, TNFA_NFKB, INTERFERON_GAMMA, WNT, etc.),
        # promote that canonical label over the medoid-derived one. This
        # keeps named minority signals visible even when they're absorbed
        # into a cluster whose medoid points elsewhere.
        member_terms_scores = [
            (recs[i].term, recs[i].nes) for i in member_idx
        ]
        anchor_term, anchor_label = _anchor_label(member_terms_scores)
        if anchor_term is not None and anchor_term != rep_rec.term:
            # anchor term is present but wasn't the medoid; override label
            final_label = anchor_label
            # keep representative_term as the medoid for auditability, but
            # flag the anchor match in the label itself so it's visible.
        elif anchor_term is not None:
            # medoid IS the anchor match — use the clean anchor label anyway
            final_label = anchor_label
        else:
            final_label = clean_label(rep_rec.term)

        themes.append(
            ClusteredTheme(
                cluster=ci,
                label=final_label,
                representative_term=rep_rec.term,
                n_members=len(member_idx),
                mean_nes=float(np.mean([recs[i].nes for i in member_idx])),
                min_q_value=float(np.min([recs[i].q_value for i in member_idx])),
                mean_intra_similarity=mean_intra,
                members=members,
                member_indices=tuple(member_idx),
            )
        )

    for i, c in enumerate(labels):
        if int(c) != 0:
            continue
        orphans.append(
            ClusteredPathway(
                term=recs[i].term,
                nes=recs[i].nes,
                q_value=recs[i].q_value,
                size=recs[i].size,
                leading_edge_n=len(gene_sets[i]),
                cluster=0,
                centrality=0.0,
                is_representative=False,
            )
        )
    orphans.sort(key=lambda o: o.q_value)

    # Renumber clusters contiguously by size desc / q-value asc
    themes.sort(key=lambda t: (-t.n_members, t.min_q_value))
    remap = {t.cluster: i + 1 for i, t in enumerate(themes)}
    new_labels = np.zeros_like(labels)
    for t in themes:
        new_id = remap[t.cluster]
        for m in t.members:
            m.cluster = new_id
        t.cluster = new_id
    for i, c in enumerate(labels):
        if int(c) in remap:
            new_labels[i] = remap[int(c)]

    return ClusteringResult(
        themes=themes, orphans=orphans, params=params,
        similarity_matrix=sim, records=list(recs),
        cluster_labels=new_labels,
    )
