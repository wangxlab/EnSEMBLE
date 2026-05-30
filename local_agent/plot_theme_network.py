"""Network plot for clustered themes (inside the v6-style agent)."""

from __future__ import annotations

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import ConvexHull

from .cluster_themes import ClusteringResult


_PREFIX_RE = re.compile(
    r"^(REACTOME_|GOBP_|GOCC_|GOMF_|HALLMARK_|KEGG_|WP_|PID_|BIOCARTA_)"
)


def _short(term: str, maxlen: int = 38) -> str:
    cleaned = _PREFIX_RE.sub("", term)
    cleaned = cleaned.replace("_", " ").title()
    if len(cleaned) > maxlen:
        cleaned = cleaned[: maxlen - 1].rstrip() + "\u2026"
    return cleaned


def plot_clustered_network(
    result: ClusteringResult,
    *,
    edge_threshold: float = 0.25,
    out_png: str | None = None,
    out_pdf: str | None = None,
    title: str | None = None,
    figsize: tuple = (22, 17),
    seed: int = 1,
) -> plt.Figure:
    recs = result.records
    sim = result.similarity_matrix
    labels = result.cluster_labels
    n = len(recs)
    if n == 0:
        raise ValueError("No pathways in clustering result.")

    nes = np.array([r.nes for r in recs])
    G = nx.Graph()
    for i, rec in enumerate(recs):
        G.add_node(i, label=rec.term, cluster=int(labels[i]),
                   nes=float(rec.nes), qval=float(rec.q_value))
    for i in range(n):
        for j in range(i + 1, n):
            w = float(sim[i, j])
            if w >= edge_threshold:
                G.add_edge(i, j, weight=w)

    rng = np.random.default_rng(seed)
    unique_clusters = sorted({int(c) for c in labels})
    sub_positions: dict = {}
    sizes = [max(1, int(np.sum(labels == c))) for c in unique_clusters]
    radii = [0.18 + 0.18 * np.sqrt(s) for s in sizes]
    for idx, c in enumerate(unique_clusters):
        members = [i for i in range(n) if int(labels[i]) == c]
        sub = G.subgraph(members).copy()
        if len(sub) == 1:
            sub_positions[c] = {members[0]: np.array([0.0, 0.0])}
        else:
            p = nx.spring_layout(
                sub, weight="weight", seed=seed + idx,
                k=1.8 / np.sqrt(len(sub)), iterations=300,
            )
            arr = np.array(list(p.values()))
            arr -= arr.mean(axis=0)
            m = np.max(np.linalg.norm(arr, axis=1))
            if m > 0:
                arr /= m
            arr *= radii[idx]
            sub_positions[c] = {v: arr[i] for i, v in enumerate(p.keys())}

    K = len(unique_clusters)
    ring_r = max(1.0, 0.9 * sum(radii) / np.pi) * 1.4
    centroids = {}
    for idx, c in enumerate(unique_clusters):
        theta = 2 * np.pi * idx / max(1, K) + rng.uniform(-0.05, 0.05)
        centroids[c] = np.array([ring_r * np.cos(theta), ring_r * np.sin(theta)])

    pos = {}
    for c in unique_clusters:
        for v, p in sub_positions[c].items():
            pos[v] = p + centroids[c]
    _parr = np.array(list(pos.values()))
    _pmax = np.max(np.abs(_parr)) if _parr.size else 1.0
    if _pmax > 0:
        for v in pos:
            pos[v] = pos[v] / _pmax

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)

    cmap = plt.get_cmap("tab20")
    cluster_colors = {}
    non_zero = [c for c in unique_clusters if c != 0]
    for i, c in enumerate(non_zero):
        cluster_colors[c] = cmap(i % 20)
    if 0 in unique_clusters:
        cluster_colors[0] = (0.6, 0.6, 0.6, 1.0)

    for c in unique_clusters:
        members = [i for i in range(n) if int(labels[i]) == c]
        pts = np.array([pos[v] for v in members])
        color = cluster_colors[c]
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                hp = pts[hull.vertices]
                centroid = hp.mean(axis=0)
                expanded = centroid + (hp - centroid) * 1.35
                ax.add_patch(plt.Polygon(
                    expanded, closed=True, facecolor=color, edgecolor=color,
                    alpha=0.18, linewidth=2, zorder=0,
                ))
            except Exception:
                pass
        elif len(pts) == 2:
            centroid = pts.mean(axis=0)
            r = np.linalg.norm(pts[0] - pts[1]) / 2 * 1.5 + 0.08
            ax.add_patch(plt.Circle(centroid, r, facecolor=color,
                                    edgecolor=color, alpha=0.18,
                                    linewidth=2, zorder=0))
        elif len(pts) == 1:
            ax.add_patch(plt.Circle(pts[0], 0.10, facecolor=color,
                                    edgecolor=color, alpha=0.18,
                                    linewidth=2, zorder=0))

        if c == 0:
            label_text = "Orphans"
        else:
            matching = next((t for t in result.themes if t.cluster == c), None)
            label_text = f"C{c}: {matching.label}" if matching else f"C{c}"
        top_y = pts[:, 1].max() if len(pts) else 0.0
        center_x = pts[:, 0].mean() if len(pts) else 0.0
        ax.text(center_x, top_y + 0.10, label_text,
                fontsize=11, fontweight="bold", color=color,
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=color, alpha=0.85), zorder=5)

    edges = list(G.edges(data=True))
    if edges:
        ws = np.array([d["weight"] for _, _, d in edges])
        wmin, wmax = ws.min(), ws.max()
        widths = 0.3 + 2.7 * (ws - wmin) / max(1e-9, wmax - wmin)
        alphas = 0.2 + 0.6 * (ws - wmin) / max(1e-9, wmax - wmin)
        for (u, v, d), w, a in zip(edges, widths, alphas):
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color="gray", linewidth=w, alpha=a, zorder=1)

    abs_nes = np.abs(nes)
    size_min, size_max = 300, 2800
    rng_s = max(1e-9, abs_nes.max() - abs_nes.min())
    node_sizes = size_min + (abs_nes - abs_nes.min()) / rng_s * (size_max - size_min)

    reps = {t.representative_term for t in result.themes}
    xs = np.array([pos[i][0] for i in range(n)])
    ys = np.array([pos[i][1] for i in range(n)])
    node_colors = [tuple(list(cluster_colors[int(labels[i])][:3]) + [0.92])
                   for i in range(n)]
    edge_colors = ["black" if recs[i].term in reps else "#333333" for i in range(n)]
    linewidths = [2.2 if recs[i].term in reps else 0.8 for i in range(n)]
    ax.scatter(xs, ys, s=node_sizes, c=node_colors, edgecolors=edge_colors,
               linewidths=linewidths, zorder=3)

    for i in range(n):
        offset = -(0.018 + 0.0009 * np.sqrt(node_sizes[i]))
        ax.text(xs[i], ys[i] + offset, _short(recs[i].term),
                fontsize=7, ha="center", va="top", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.7))

    cluster_handles = []
    for c in non_zero:
        theme = next((t for t in result.themes if t.cluster == c), None)
        name = f"C{c}: {theme.label}" if theme else f"C{c}"
        cluster_handles.append(mpatches.Patch(
            facecolor=cluster_colors[c], edgecolor="black",
            label=f"{name} (n={int(np.sum(labels == c))})"))
    if 0 in unique_clusters:
        cluster_handles.append(mpatches.Patch(
            facecolor=cluster_colors[0], edgecolor="black",
            label=f"Orphans (n={int(np.sum(labels == 0))})"))
    leg1 = ax.legend(handles=cluster_handles, loc="upper left",
                     title="Clusters (representative term)",
                     fontsize=8, title_fontsize=10, frameon=True)
    ax.add_artist(leg1)

    sz_vals = np.linspace(abs_nes.min(), abs_nes.max(), 3)
    sz_handles = []
    for v in sz_vals:
        t = (v - abs_nes.min()) / rng_s
        s = size_min + t * (size_max - size_min)
        sz_handles.append(plt.scatter([], [], s=s, facecolor="lightgray",
                                      edgecolor="black",
                                      label=f"|NES|={v:.2f}"))
    leg2 = ax.legend(handles=sz_handles, loc="upper right",
                     title="Node size",
                     fontsize=9, title_fontsize=10, labelspacing=1.5,
                     borderpad=1.0, frameon=True, scatterpoints=1)
    ax.add_artist(leg2)

    rep_handles = [
        plt.scatter([], [], s=260, facecolor="white",
                    edgecolor="black", linewidths=2.2,
                    label="Representative (medoid)"),
        plt.scatter([], [], s=260, facecolor="white",
                    edgecolor="#333333", linewidths=0.8,
                    label="Other member"),
    ]
    leg3 = ax.legend(handles=rep_handles, loc="lower right",
                     title="Node border", fontsize=9, title_fontsize=10,
                     frameon=True, scatterpoints=1)
    ax.add_artist(leg3)

    ax.set_title(
        title or "GSEA themes - leading-edge Jaccard + dynamicTreeCut",
        fontsize=14, pad=18,
    )

    plt.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    return fig
