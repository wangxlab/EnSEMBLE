"""Three deterministic figure generators per report_module_spec.md.

All figures: matplotlib only, no LLM. Outputs both .pdf (archival) and .png
(for HTML/weasyprint embed) in the same directory.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")  # headless; safe on HPC

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

# Color palette (matches spec)
COLOR_SUPPORTED = "#3B6D11"
COLOR_PARTIAL = "#854F0B"
COLOR_GLO = "#B4B2A9"
COLOR_UP = "#B71C1C"
COLOR_DOWN = "#1976D2"
COLOR_CELLTYPE = "#1E88E5"
COLOR_TFBS = "#FB8C00"


def _save_both(fig: plt.Figure, output_path: Path) -> None:
    """Save the figure as both PDF (archival) and PNG (HTML embed)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------- Figure 1: Compression Summary ----------


def plot_compression(
    n_gene_sets: int,
    n_themes: int,
    n_supported: int,
    n_partial: int,
    n_glo: int,
    output_path: Path,
    title: str = "Compression Summary",
) -> None:
    """Horizontal stacked bar with separate above-bar labels when segments are tiny."""
    fig, ax = plt.subplots(figsize=(9, 3.6))

    bar_height = 0.55
    max_n = max(n_gene_sets, n_themes, n_supported + n_partial + n_glo, 1)
    label_offset = max_n * 0.012

    # Row 1: significant gene sets
    ax.barh(2, n_gene_sets, height=bar_height, color="#5C6BC0", edgecolor="white")
    ax.text(n_gene_sets + label_offset, 2, str(n_gene_sets), va="center", fontsize=10)

    # Row 2: themes
    ax.barh(1, n_themes, height=bar_height, color="#26A69A", edgecolor="white")
    ax.text(n_themes + label_offset, 1, str(n_themes), va="center", fontsize=10)

    # Row 3: verdicts (stacked)
    x = 0
    segments = [
        (n_supported, COLOR_SUPPORTED, "SUPPORTED"),
        (n_partial, COLOR_PARTIAL, "PARTIAL"),
        (n_glo, COLOR_GLO, "GLO"),
    ]
    for value, color, _ in segments:
        if value > 0:
            ax.barh(0, value, left=x, height=bar_height, color=color, edgecolor="white", linewidth=0.6)
            x += value
    # Compose verdict label as one string after the bar
    parts = []
    if n_supported:
        parts.append(f"{n_supported} SUP")
    if n_partial:
        parts.append(f"{n_partial} PAR")
    if n_glo:
        parts.append(f"{n_glo} GLO")
    ax.text(x + label_offset, 0, " · ".join(parts), va="center", fontsize=10)

    # Row labels (left of bars)
    ax.text(-max_n * 0.02, 2, "Significant\ngene sets", ha="right", va="center", fontsize=10)
    ax.text(-max_n * 0.02, 1, "Themes\n(clustered)", ha="right", va="center", fontsize=10)
    ax.text(-max_n * 0.02, 0, "Verdicts", ha="right", va="center", fontsize=10)

    # Reductions
    if n_gene_sets > 0:
        reduction1 = 100.0 * (1 - n_themes / n_gene_sets)
        ax.text(
            n_themes + label_offset * 4,
            1.5,
            f"clustering: -{reduction1:.0f}%",
            ha="left",
            va="center",
            fontsize=9,
            color="#666",
            style="italic",
        )
    if n_themes > 0:
        sp = n_supported + n_partial
        reduction2 = 100.0 * (1 - sp / n_themes)
        ax.text(
            x + label_offset * 4 + max_n * 0.05,
            0.5,
            f"agent filter: -{reduction2:.0f}% (kept {sp}/{n_themes})",
            ha="left",
            va="center",
            fontsize=9,
            color="#666",
            style="italic",
        )

    ax.set_xlim(-max_n * 0.22, max_n * 1.30)
    ax.set_ylim(-0.55, 2.55)
    ax.set_yticks([])
    ax.set_xlabel("Count")
    ax.set_title(title, loc="left", fontsize=12, weight="bold")

    # Legend for verdict colors
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=COLOR_SUPPORTED, label="SUPPORTED"),
        plt.Rectangle((0, 0), 1, 1, fc=COLOR_PARTIAL, label="PARTIAL"),
        plt.Rectangle((0, 0), 1, 1, fc=COLOR_GLO, label="GENE_LEVEL_ONLY"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9)

    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)

    _save_both(fig, output_path)


# ---------- Figure 2: Evidence Network ----------


def plot_evidence_network(
    verdicts: List[dict],
    helpers: List[dict],
    themes: List[dict],
    output_path: Path,
    title: str = "Evidence Network",
    n_glo_to_show: int = 10,
) -> None:
    """Bipartite helper <-> theme network with GLO context (v2.1 spec).

    Right column shows two zones:
      - Top: SUPPORTED/PARTIAL themes with edges to helpers, sorted by
        theme_weight desc.
      - Bottom: top GLO themes (no edges, dashed border, sorted by |NES| desc).

    Node sizing:
      - Helpers: by -log10(q-value)
      - SP themes: by theme_weight (Stouffer-like)
      - GLO themes: by |mean_nes|

    Edge styling:
      - SUPPORTED: solid green
      - PARTIAL: thinner brown
      - thickness scales with -log10(helper_q)
    """
    helpers_by_name = {h["helper_name"]: h for h in helpers}
    themes_by_id = {t["theme_id"]: t for t in themes}

    sp_verdicts = [v for v in verdicts if v["verdict"] in ("SUPPORTED", "PARTIAL")]
    glo_verdicts = [v for v in verdicts if v["verdict"] == "GENE_LEVEL_ONLY"]

    used_helpers: set[str] = set()
    used_sp_themes: set[str] = set()
    edges: list[tuple] = []
    for v in sp_verdicts:
        link_list = v.get("linked_helpers") or (
            [v["linked_helper"]] if v.get("linked_helper") else []
        )
        if not link_list:
            continue
        used_sp_themes.add(v["theme_id"])
        for h in link_list:
            used_helpers.add(h)
            edges.append((h, v["theme_id"], v["verdict"]))

    # Sort helpers by q-value (most significant top)
    helper_list = sorted(
        used_helpers, key=lambda h: helpers_by_name.get(h, {}).get("q_value", 1.0)
    )

    # SP theme list: sort by theme_weight desc (fall back to |NES|)
    def _sp_weight(tid: str) -> float:
        for v in sp_verdicts:
            if v["theme_id"] == tid:
                tw = v.get("theme_weight")
                if isinstance(tw, (int, float)):
                    return tw
        meta = themes_by_id.get(tid, {})
        return abs(meta.get("mean_nes", 0))

    sp_theme_list = sorted(used_sp_themes, key=lambda t: -_sp_weight(t))

    # GLO theme list: top N by |NES|
    glo_with_nes = []
    for v in glo_verdicts:
        meta = themes_by_id.get(v["theme_id"], {})
        nes = meta.get("mean_nes", 0)
        glo_with_nes.append((v["theme_id"], abs(nes)))
    glo_with_nes.sort(key=lambda x: -x[1])
    glo_theme_list = [tid for tid, _ in glo_with_nes[:n_glo_to_show]]
    n_glo_total = len(glo_verdicts)
    n_glo_shown = len(glo_theme_list)

    if not sp_theme_list and not glo_theme_list:
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.text(
            0.5, 0.5,
            "No themes to display.",
            ha="center", va="center", fontsize=12, color="#555",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        ax.set_title(title, loc="left", fontsize=12, weight="bold")
        _save_both(fig, output_path)
        return

    # Layout: helpers on left at x=0, themes on right at x=1
    # Right column splits vertically: SP zone (top) + GLO zone (bottom).
    # Allocate a small gap between zones.
    n_h = max(1, len(helper_list))
    n_sp = len(sp_theme_list)
    n_glo = len(glo_theme_list)
    # Total right-column slots = n_sp + (gap=1) + n_glo
    n_right_total = n_sp + (1 if (n_sp and n_glo) else 0) + n_glo
    rows = max(n_h, n_right_total, 1)

    pos: dict = {}
    # Helpers: even spacing
    h_step = rows / max(1, n_h)
    for i, h in enumerate(helper_list):
        pos[h] = (0.0, rows - (i + 0.5) * h_step)

    # Right column: SP themes top, GLO bottom, separator gap of ~1 row
    if n_right_total > 0:
        right_step = rows / max(1, n_right_total)
        cursor = 0
        for tid in sp_theme_list:
            pos[tid] = (1.0, rows - (cursor + 0.5) * right_step)
            cursor += 1
        if n_sp and n_glo:
            cursor += 1  # gap
        for tid in glo_theme_list:
            pos[tid] = (1.0, rows - (cursor + 0.5) * right_step)
            cursor += 1

    fig_h = max(4.5, 0.42 * rows + 1.8)
    fig, ax = plt.subplots(figsize=(14, fig_h))

    # Edges (under nodes)
    for u, w, verdict in edges:
        q = helpers_by_name.get(u, {}).get("q_value", 1.0)
        alpha = 0.35 + 0.55 * min(1.0, -math.log10(max(q, 1e-30)) / 30)
        color = COLOR_SUPPORTED if verdict == "SUPPORTED" else COLOR_PARTIAL
        # Edge thickness scales with -log10(q); bounded.
        lw_base = 1.0 + 0.10 * min(-math.log10(max(q, 1e-30)), 30)
        lw = lw_base * (1.0 if verdict == "SUPPORTED" else 0.7)
        if u in pos and w in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[w]
            ax.plot([x0, x1], [y0, y1], color=color, alpha=alpha, lw=lw, zorder=1)

    # Helper nodes
    for h in helper_list:
        meta = helpers_by_name.get(h, {})
        cls = meta.get("helper_class", "tfbs")
        color = COLOR_CELLTYPE if cls == "celltype" else COLOR_TFBS
        q = meta.get("q_value", 1.0)
        size = 220 + 30 * min(-math.log10(max(q, 1e-30)), 30)
        x, y = pos[h]
        ax.scatter(x, y, s=size, c=color, edgecolors="black", linewidths=0.7, zorder=3)
        ax.text(x - 0.04, y, h, ha="right", va="center", fontsize=9)

    # SP theme nodes (solid border, sized by theme_weight)
    for tid in sp_theme_list:
        meta = themes_by_id.get(tid, {})
        direction = meta.get("direction", "UP")
        color = COLOR_UP if direction == "UP" else COLOR_DOWN
        weight = _sp_weight(tid)
        size = 240 + 8 * min(weight, 100)  # theme_weight can be very large; bound
        x, y = pos[tid]
        ax.scatter(x, y, s=size, c=color, edgecolors="black", linewidths=0.8, zorder=3)
        label = meta.get("label", tid)[:40]
        ax.text(x + 0.04, y, label, ha="left", va="center", fontsize=9)

    # GLO theme nodes (dashed border, sized by |mean_nes|)
    for tid in glo_theme_list:
        meta = themes_by_id.get(tid, {})
        direction = meta.get("direction", "UP")
        # Use a faded version (colored fill, lighter alpha)
        face_color = COLOR_UP if direction == "UP" else COLOR_DOWN
        nes = abs(meta.get("mean_nes", 0))
        size = 200 + 200 * nes
        x, y = pos[tid]
        ax.scatter(
            x, y, s=size, c=face_color, edgecolors="black",
            linewidths=1.0, linestyle="--", alpha=0.45, zorder=3,
        )
        label = meta.get("label", tid)[:40]
        ax.text(
            x + 0.04, y, label, ha="left", va="center",
            fontsize=8.5, color="#444", style="italic",
        )

    # GLO truncation note
    if n_glo_shown < n_glo_total:
        ax.text(
            1.0,
            (rows - n_right_total * (rows / max(1, n_right_total))) - rows * 0.03,
            f"+ {n_glo_total - n_glo_shown} more GLO themes (not shown)",
            ha="left", va="top", fontsize=8, style="italic", color="#888",
        )

    # Zone separator label between SP and GLO
    if sp_theme_list and glo_theme_list:
        sep_y = pos[glo_theme_list[0]][1] + (rows / max(1, n_right_total)) * 0.6
        ax.axhline(sep_y, xmin=0.42, xmax=0.92, color="#bbb", lw=0.5, linestyle=":", zorder=0)
        ax.text(
            1.0, sep_y + 0.05, "─ GLO themes (no enhancer support)",
            ha="left", va="bottom", fontsize=8, style="italic", color="#888",
        )

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_CELLTYPE,
                   markeredgecolor="black", markersize=10, label="Helper: cell-type"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_TFBS,
                   markeredgecolor="black", markersize=10, label="Helper: TFBS"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_UP,
                   markeredgecolor="black", markersize=10, label="Theme: UP"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_DOWN,
                   markeredgecolor="black", markersize=10, label="Theme: DOWN"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=(0.7, 0.7, 0.7, 0.45),
                   markeredgecolor="black", linestyle="--", markersize=10, label="GLO theme"),
        plt.Line2D([0], [0], color=COLOR_SUPPORTED, lw=2, label="SUPPORTED edge"),
        plt.Line2D([0], [0], color=COLOR_PARTIAL, lw=1.2, label="PARTIAL edge"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=8, frameon=False)

    ax.set_xlim(-1.4, 2.4)
    ax.set_ylim(-0.6, rows + 0.5)
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=12, weight="bold")

    _save_both(fig, output_path)


# ---------- Figure 3: ESEA Helper Overview ----------


def plot_helper_overview(
    helpers: List[dict],
    verdicts: List[dict],
    output_path: Path,
    title: str = "ESEA Helper Overview",
) -> None:
    """Horizontal lollipop. Y-axis: helper names. X-axis: NES.

    Filled dot = helper used in >=1 SUPPORTED/PARTIAL verdict.
    Open dot   = significant but not linked.
    Right-margin annotation lists the linked theme(s) for filled dots.
    """
    if not helpers:
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.text(0.5, 0.5, "No helpers in input.", ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_axis_off()
        _save_both(fig, output_path)
        return

    used_helpers: dict[str, list[str]] = {}
    for v in verdicts:
        if v["verdict"] not in ("SUPPORTED", "PARTIAL"):
            continue
        link_list = v.get("linked_helpers") or (
            [v["linked_helper"]] if v.get("linked_helper") else []
        )
        for h in link_list:
            used_helpers.setdefault(h, []).append(v["theme_label"])

    sorted_helpers = sorted(helpers, key=lambda h: -abs(h["nes"]))
    n = len(sorted_helpers)
    fig_h = max(4.0, 0.32 * n + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    # Compute x range so we have room for right-margin annotation
    nes_vals = [h["nes"] for h in sorted_helpers]
    x_min = min(nes_vals + [0])
    x_max = max(nes_vals + [0])
    span = max(0.1, x_max - x_min)
    annotation_x = x_max + span * 0.15

    yticks = list(range(n))
    yticklabels = [h["helper_name"] for h in sorted_helpers]

    for i, h in enumerate(sorted_helpers):
        y = n - 1 - i  # top-to-bottom by descending |NES|
        nes = h["nes"]
        color = COLOR_DOWN if nes < 0 else COLOR_UP
        is_linked = h["helper_name"] in used_helpers
        # Stem
        ax.plot([0, nes], [y, y], color=color, lw=1.0, alpha=0.45)
        # Dot
        if is_linked:
            ax.scatter(nes, y, s=80, facecolor=color, edgecolor="black", linewidths=0.6, zorder=3)
        else:
            ax.scatter(nes, y, s=80, facecolor="white", edgecolor=color, linewidths=1.4, zorder=3)
        # Linked themes annotation, in the right margin
        if is_linked:
            theme_str = "; ".join(t[:32] for t in used_helpers[h["helper_name"]])
            ax.text(annotation_x, y, "-> " + theme_str, ha="left", va="center", fontsize=7.5, style="italic", color="#444")

    # Helper names on Y-axis (so they never overlap dots/stems)
    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels(yticklabels, fontsize=8.5)
    ax.axvline(0, color="black", lw=0.6, zorder=1)
    ax.set_xlabel("Helper NES (DOWN | UP)")
    ax.set_xlim(x_min - span * 0.05, annotation_x + span * 0.55)
    ax.set_title(title, loc="left", fontsize=12, weight="bold")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_DOWN, markeredgecolor="black", markersize=9, label="DOWN, linked"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor=COLOR_DOWN, markersize=9, label="DOWN, unlinked"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_UP, markeredgecolor="black", markersize=9, label="UP, linked"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor=COLOR_UP, markersize=9, label="UP, unlinked"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=4, fontsize=8, frameon=False)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    _save_both(fig, output_path)
