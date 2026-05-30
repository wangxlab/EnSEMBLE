"""Build a single PDF report for a dataset using v2.0 consensus verdicts.

Usage:
    python -m local_agent.report.build_report --dataset bt20
    python -m local_agent.report.build_report --dataset all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..config import OUTPUTS_DIR, AgentConfig, INPUTS_DIR
from ..runner import DATASET_NAMES
from .pdf_assembler import assemble_report
from .figures import (
    plot_compression,
    plot_evidence_network,
    plot_helper_overview,
)
from .verdict_table import write_verdict_table


V21_DIR = OUTPUTS_DIR / "v2_2_lock"


def _count_significant_gene_sets(dataset: str, q_threshold: float) -> int:
    """Count significant gene sets in the raw GSEA CSV (q < threshold)."""
    csv = INPUTS_DIR / dataset / "GSEA_results.csv"
    if not csv.exists():
        return 0
    df = pd.read_csv(csv)
    return int((df["qValue"] < q_threshold).sum())


def build_one_report(dataset: str, lock_dir: Path = V21_DIR) -> Path:
    ds_dir = lock_dir / dataset
    verdicts_path = ds_dir / "verdicts.json"
    input_path = ds_dir / "agent_input.json"
    if not verdicts_path.exists():
        raise FileNotFoundError(f"Missing v2.0 verdicts at {verdicts_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing agent_input at {input_path}")

    verdicts = json.loads(verdicts_path.read_text())["verdicts"]
    inp = json.loads(input_path.read_text())

    helpers = inp["helpers"]
    themes = inp["themes"]

    # Counts
    counts = {"SUPPORTED": 0, "PARTIAL": 0, "GENE_LEVEL_ONLY": 0}
    for v in verdicts:
        counts[v["verdict"]] += 1

    config = AgentConfig()
    n_gene_sets = _count_significant_gene_sets(dataset, config.gsea_q_threshold)

    figures_dir = ds_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_compression(
        n_gene_sets=n_gene_sets,
        n_themes=len(themes),
        n_supported=counts["SUPPORTED"],
        n_partial=counts["PARTIAL"],
        n_glo=counts["GENE_LEVEL_ONLY"],
        output_path=figures_dir / "fig_compression",
        title=f"Compression Summary — {dataset.upper()}",
    )
    plot_evidence_network(
        verdicts=verdicts,
        helpers=helpers,
        themes=themes,
        output_path=figures_dir / "fig_network",
        title=f"Evidence Network — {dataset.upper()}",
    )
    plot_helper_overview(
        helpers=helpers,
        verdicts=verdicts,
        output_path=figures_dir / "fig_esea_overview",
        title=f"ESEA Helper Overview — {dataset.upper()}",
    )

    # Verdict table
    table_path = ds_dir / "verdict_table.md"
    write_verdict_table(verdicts, themes, table_path)

    # Mini-thesis (optional; only if file exists)
    thesis_path = ds_dir / "mini_thesis.md"
    if not thesis_path.exists():
        thesis_path = None

    # Parameters footer
    params_line = (
        f"Parameters: q={config.gsea_q_threshold}, "
        f"model={config.model}, temperature={config.temperature}, "
        f"max_tokens={config.max_tokens}, "
        f"capacity_cap=3 per helper, "
        f"clustering=two-bin auto (locked)"
    )

    pdf_path = ds_dir / f"report_{dataset}.pdf"
    assemble_report(
        dataset_id=dataset,
        figures_dir=figures_dir,
        mini_thesis_path=thesis_path,
        verdict_table_md_path=table_path,
        output_pdf_path=pdf_path,
        parameters_line=params_line,
    )
    return pdf_path


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--lock-dir",
        type=Path,
        default=V21_DIR,
        help=f"v2.0 lock directory (default {V21_DIR})",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    targets = list(DATASET_NAMES) if args.dataset == "all" else [args.dataset]
    for ds in targets:
        print(f"\n=== Building report for {ds} ===")
        pdf = build_one_report(ds, args.lock_dir)
        print(f"  -> {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
