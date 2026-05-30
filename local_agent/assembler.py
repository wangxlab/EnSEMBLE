"""Assemble the AgentInput JSON from per-dataset raw files.

Pipeline:
  1. Load GSEA_results.csv -> prefilter -> records (UP/DOWN).
  2. Run build_theme_summaries (locked clustering pipeline) -> ThemeSummary list.
  3. Adapt ThemeSummary -> agent Theme schema.
  4. Parse ESEA_helpers.csv -> agent Helper schema.
  5. Read backgrounds.txt -> biological_context.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd

from .merger import (
    MergeLog,
    merge_redundant_themes,
    render_merge_log_markdown,
)
from .config import AnalysisSettings
from .prefilter import (
    load_gsea,
    split_gsea_by_direction,
    to_gsea_records,
)
from .schemas import AgentInput, Helper, Theme
from .themes import build_theme_summaries


_CELLTYPE_PREFIXES = ("ENCODE_", "CATlas_", "eRNAbase_", "dbSUPER_")


def classify_helper(name: str) -> str:
    """Heuristic class label for the helper.

    Cell-type enhancer-program references typically follow naming conventions
    (ENCODE_*, CATlas_*, eRNAbase_*, dbSUPER_*). Anything else is treated as
    a TFBS-style helper (gene-symbol named, e.g. POLR2A, EP300, SUPT5H).
    """
    if name.startswith(_CELLTYPE_PREFIXES):
        return "celltype"
    return "tfbs"


def parse_helpers(esea_csv: Path) -> List[Helper]:
    df = pd.read_csv(esea_csv)
    required = {"Compare.List", "NES", "qValue", "leadingEdge"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{esea_csv}: missing helper columns {missing}")

    helpers: List[Helper] = []
    for _, row in df.iterrows():
        name = str(row["Compare.List"])
        nes = float(row["NES"])
        direction = "UP" if nes > 0 else "DOWN"
        leading_edge = str(row["leadingEdge"]) if pd.notna(row["leadingEdge"]) else ""
        leading_edge_n = (
            len([x for x in leading_edge.split(",") if x.strip()])
            if leading_edge
            else 0
        )
        top_hallmark = row.get("top_hallmark")
        if pd.isna(top_hallmark) or str(top_hallmark).strip().lower() in (
            "",
            "no significant hallmark annotation",
        ):
            top_hallmark_clean = None
        else:
            top_hallmark_clean = str(top_hallmark)

        helpers.append(
            Helper(
                helper_name=name,
                helper_class=classify_helper(name),
                direction=direction,
                nes=nes,
                q_value=float(row["qValue"]),
                leading_edge_n=leading_edge_n,
                top_hallmark=top_hallmark_clean,
            )
        )
    return helpers


def cluster_themes(
    gsea_csv: Path,
    output_dir: Path,
    q_threshold: float = 0.05,
    theme_cap_per_direction: int = 0,
    theme_cap_total: int = 40,
) -> List[Theme]:
    """Run the locked clustering pipeline and adapt to agent Theme schema.

    Side effects: writes cluster_themes_*.json + network plots into output_dir
    (consistent with the upstream pipeline). Clustering parameters are locked
    via cluster_auto_params=True (default in AnalysisSettings); only the
    downstream theme caps are exposed here so we don't drop biologically
    critical themes from the agent's input.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # AnalysisSettings: theme_cap=0 means unlimited; theme_cap_total caps total.
    # Upstream uses 1_000_000 as the "unlimited" sentinel for theme_cap_total,
    # so we pass it through unchanged when caller wants no global cap.
    settings = AnalysisSettings(
        gsea_q_threshold=q_threshold,
        theme_cap=theme_cap_per_direction,
        theme_cap_total=theme_cap_total if theme_cap_total > 0 else 1_000_000,
    )

    rows = load_gsea(gsea_csv)
    direction_tables = split_gsea_by_direction(rows, settings)
    gsea_records = {
        d: to_gsea_records(rws, d) for d, rws in direction_tables.items()
    }

    themes_by_dir = build_theme_summaries(gsea_records, settings, output_dir=output_dir)

    adapted: List[Theme] = []
    for direction in ("UP", "DOWN"):
        for t in themes_by_dir.get(direction, []):
            top_pathways = [r.term for r in t.top_pathways]
            adapted.append(
                Theme(
                    theme_id=t.theme_id,
                    label=t.label,
                    direction=t.direction,
                    mean_nes=float(t.effect),
                    n_members=max(1, len(t.top_pathways)),
                    top_pathways=top_pathways,
                    top_leading_edge_genes=list(t.leading_edges),
                )
            )
    return adapted


def _build_full_le_lookup(
    themes: List[Theme],
    gsea_csv: Path,
    clustering_dir: Path,
) -> dict:
    """Build {theme_id: set(genes)} containing the union of every member
    pathway's leading edge for that theme.

    Reads cluster_themes_<dir>.json (written by build_theme_summaries) to
    find which raw pathway terms each theme aggregates, then looks each term
    up in the original GSEA CSV to grab its full leading edge.
    """
    df = pd.read_csv(gsea_csv)
    le_per_term: dict = {}
    for _, row in df.iterrows():
        term = str(row["Compare.List"])
        le_str = row.get("leadingEdge")
        if pd.isna(le_str) or not le_str:
            le_per_term[term] = set()
            continue
        le_per_term[term] = {g.strip() for g in str(le_str).split(",") if g.strip()}

    # Index cluster_themes_<dir>.json contents by direction
    members_per_label_per_dir: dict[str, dict[str, list[str]]] = {"UP": {}, "DOWN": {}}
    orphan_term_per_dir: dict[str, set[str]] = {"UP": set(), "DOWN": set()}
    for direction in ("UP", "DOWN"):
        path = clustering_dir / f"cluster_themes_{direction.lower()}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for c in data.get("themes", []):
            label = c.get("label")
            if not label:
                continue
            members_per_label_per_dir[direction][label] = [
                m.get("term") for m in c.get("members", []) if m.get("term")
            ]
        for o in data.get("orphans", []):
            term = o.get("term")
            if term:
                orphan_term_per_dir[direction].add(term)

    out: dict = {}
    for t in themes:
        terms: list[str] = []
        # Real cluster: match by label
        terms = list(members_per_label_per_dir.get(t.direction, {}).get(t.label, []))
        if not terms:
            # Orphan or unmatched -> use top_pathways[0] (the single representative term)
            for cand in t.top_pathways or []:
                if cand in le_per_term:
                    terms.append(cand)
                    break
        union: set = set()
        for term in terms:
            union |= le_per_term.get(term, set())
        if union:
            out[t.theme_id] = union
    return out


def assemble_input(
    dataset_id: str,
    dataset_dir: Path,
    output_dir: Path,
    q_threshold: float = 0.05,
    theme_cap_per_direction: int = 0,
    theme_cap_total: int = 40,
    merge_jaccard_threshold: float = 0.5,
) -> AgentInput:
    """Convention-driven: expects <dataset_dir>/{GSEA_results.csv,
    ESEA_helpers.csv, backgrounds.txt}.

    For explicit file paths (e.g. the v1.x CLI shim), use
    ``assemble_input_from_files``.
    """
    return assemble_input_from_files(
        dataset_id=dataset_id,
        gsea_csv=dataset_dir / "GSEA_results.csv",
        esea_csv=dataset_dir / "ESEA_helpers.csv",
        background_txt=dataset_dir / "backgrounds.txt",
        output_dir=output_dir,
        q_threshold=q_threshold,
        theme_cap_per_direction=theme_cap_per_direction,
        theme_cap_total=theme_cap_total,
        merge_jaccard_threshold=merge_jaccard_threshold,
    )


def assemble_input_from_files(
    dataset_id: str,
    gsea_csv: Path,
    esea_csv: Path,
    background_txt: Path,
    output_dir: Path,
    q_threshold: float = 0.05,
    theme_cap_per_direction: int = 0,
    theme_cap_total: int = 40,
    merge_jaccard_threshold: float = 0.5,
) -> AgentInput:
    """Assemble agent input from explicit file paths."""
    gsea_path = Path(gsea_csv)
    helpers_path = Path(esea_csv)
    background_path = Path(background_txt)

    for p in (gsea_path, helpers_path, background_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    biological_context = background_path.read_text().strip()
    helpers = parse_helpers(helpers_path)
    themes = cluster_themes(
        gsea_path,
        output_dir / "clustering",
        q_threshold,
        theme_cap_per_direction=theme_cap_per_direction,
        theme_cap_total=theme_cap_total,
    )

    # v2.2.1: post-clustering merger to fold near-redundant themes.
    if merge_jaccard_threshold > 0:
        # Build a richer leading-edge representation by unioning every member
        # pathway's full leading edge from the GSEA CSV. The 12-gene cap on
        # Theme.top_leading_edge_genes systematically under-counts overlap
        # between biologically-related themes (e.g. iPSC's 5 cell-cycle
        # clusters share ~5-6 of their top 12 genes despite huge biological
        # overlap in their full leading edges).
        full_le_lookup = _build_full_le_lookup(
            themes, gsea_path, output_dir / "clustering"
        )
        merged_themes, merge_log = merge_redundant_themes(
            themes,
            jaccard_threshold=merge_jaccard_threshold,
            dataset_id=dataset_id,
            full_le_lookup=full_le_lookup,
        )
        # Persist the log next to clustering outputs.
        merge_dir = output_dir / "clustering"
        merge_dir.mkdir(parents=True, exist_ok=True)
        (merge_dir / "merge_log.json").write_text(
            json.dumps(merge_log.to_dict(), indent=2)
        )
        (merge_dir / "merge_log.md").write_text(
            render_merge_log_markdown(merge_log)
        )
        themes = merged_themes

    return AgentInput(
        dataset_id=dataset_id,
        biological_context=biological_context,
        helpers=helpers,
        themes=themes,
    )
