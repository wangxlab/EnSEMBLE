"""End-to-end runner.

Two invocation modes:

    # Convention-driven (recommended): expects inputs/<dataset>/{...}
    python -m local_agent.runner --dataset bt20
    python -m local_agent.runner --dataset all

    # File-path direct (also accepted by local_agent.cli shim):
    python -m local_agent.runner \\
        --gsea-csv path/to/GSEA_results.csv \\
        --esea-csv path/to/ESEA_helpers.csv \\
        --background-txt path/to/background.txt \\
        --output-dir path/to/out

Output layout per dataset:
    <output-dir>/<dataset_id>/
        agent_input.json
        verdicts.json
        validation.json
        api_log.json
        clustering/cluster_themes_{up,down}.json
        clustering/merge_log.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from .assembler import assemble_input, assemble_input_from_files
from .caller import APICallError, call_claude
from .config import INPUTS_DIR, OUTPUTS_DIR, AgentConfig
from .fallback import build_fallback_output
from .schemas import AgentInput, AgentOutput
from .validator import (
    auto_correct_capacity,
    is_capacity_only_errors,
    validate_verdicts,
)
from .weight import annotate_weights


# Default datasets used when --dataset all is passed. Any other dataset name
# is accepted too as long as inputs/<name>/{GSEA_results.csv, ESEA_helpers.csv,
# backgrounds.txt} exists.
DATASET_NAMES = ("bt20", "ipsc", "panc1", "snai1")


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def _classify_and_emit(
    agent_input: AgentInput,
    config: AgentConfig,
    out_dir: Path,
    use_api: bool,
) -> AgentOutput:
    _write_json(out_dir / "agent_input.json", agent_input.model_dump(mode="json"))

    if not use_api:
        agent_output = build_fallback_output(agent_input)
        _write_json(out_dir / "verdicts.json", agent_output.model_dump(mode="json"))
        _write_json(
            out_dir / "validation.json",
            {"errors": [], "note": "use_api=False; fallback output"},
        )
        return agent_output

    parsed, api_log = call_claude(agent_input, config)
    _write_json(out_dir / "api_log.json", api_log)

    used_fallback = False
    errors: list[str] = []
    auto_correction: dict | None = None

    if parsed is None:
        agent_output = build_fallback_output(agent_input)
        used_fallback = True
        errors.append("API call returned no parseable AgentOutput")
    else:
        errors = validate_verdicts(agent_input, parsed)
        # Soft auto-correction: capacity-cap + dedup + short-rationale fixes.
        if errors and is_capacity_only_errors(errors):
            corrected, demoted_ids = auto_correct_capacity(agent_input, parsed)
            re_errors = validate_verdicts(agent_input, corrected)
            if not re_errors:
                parsed = corrected
                auto_correction = {
                    "applied": True,
                    "demoted_theme_ids": demoted_ids,
                    "original_capacity_errors": errors,
                }
                errors = []
            else:
                errors = re_errors

        if errors:
            agent_output = build_fallback_output(agent_input)
            used_fallback = True
        else:
            agent_output = parsed

    annotate_weights(agent_input, agent_output)

    _write_json(out_dir / "verdicts.json", agent_output.model_dump(mode="json"))
    val_record: dict = {"errors": errors, "used_fallback": used_fallback}
    if auto_correction is not None:
        val_record["auto_correction"] = auto_correction
    _write_json(out_dir / "validation.json", val_record)
    return agent_output


def run_dataset(
    dataset: str,
    config: AgentConfig,
    inputs_dir: Path = INPUTS_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
    use_api: bool = True,
) -> AgentOutput:
    """Convention-driven entry: <inputs_dir>/<dataset>/* → <outputs_dir>/<dataset>/."""
    dataset_dir = inputs_dir / dataset
    out_dir = outputs_dir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    agent_input = assemble_input(
        dataset_id=dataset,
        dataset_dir=dataset_dir,
        output_dir=out_dir,
        q_threshold=config.gsea_q_threshold,
        theme_cap_per_direction=config.theme_cap_per_direction,
        theme_cap_total=config.theme_cap_total,
        merge_jaccard_threshold=config.merge_jaccard_threshold,
    )
    return _classify_and_emit(agent_input, config, out_dir, use_api)


def run_from_files(
    gsea_csv: Path,
    esea_csv: Path,
    background_txt: Path,
    config: AgentConfig,
    output_dir: Path,
    dataset_id: Optional[str] = None,
    use_api: bool = True,
) -> AgentOutput:
    """File-path direct entry. ``output_dir`` is the per-dataset folder; the
    dataset_id (used for downstream artefact naming) defaults to the parent
    directory name of gsea_csv if not supplied.
    """
    gsea_csv = Path(gsea_csv)
    esea_csv = Path(esea_csv)
    background_txt = Path(background_txt)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dataset_id is None:
        dataset_id = gsea_csv.parent.name or "dataset"

    agent_input = assemble_input_from_files(
        dataset_id=dataset_id,
        gsea_csv=gsea_csv,
        esea_csv=esea_csv,
        background_txt=background_txt,
        output_dir=output_dir,
        q_threshold=config.gsea_q_threshold,
        theme_cap_per_direction=config.theme_cap_per_direction,
        theme_cap_total=config.theme_cap_total,
        merge_jaccard_threshold=config.merge_jaccard_threshold,
    )
    return _classify_and_emit(agent_input, config, output_dir, use_api)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    defaults = AgentConfig()

    # Either --dataset OR (--gsea-csv + --esea-csv + --background-txt)
    ap.add_argument("--dataset", help="Dataset name (e.g. bt20) or 'all' for all defaults")
    ap.add_argument("--gsea-csv", type=Path, help="Path to GSEA results CSV (file-path mode)")
    ap.add_argument("--esea-csv", type=Path, help="Path to ESEA helpers CSV (file-path mode)")
    ap.add_argument("--background-txt", type=Path, help="Path to background.txt (file-path mode)")
    ap.add_argument("--inputs-dir", type=Path, default=INPUTS_DIR,
                    help="Root directory for --dataset lookup (default: inputs/)")
    ap.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR,
                    help="Output root. Each dataset writes to <output-dir>/<dataset>/")

    ap.add_argument("--model", default=defaults.model)
    ap.add_argument("--temperature", type=float, default=defaults.temperature)
    ap.add_argument("--max-tokens", type=int, default=defaults.max_tokens)
    ap.add_argument("--max-retries", type=int, default=defaults.max_retries)
    ap.add_argument("--q-threshold", type=float, default=defaults.gsea_q_threshold)
    ap.add_argument("--merge-jaccard", type=float, default=defaults.merge_jaccard_threshold,
                    help="Post-clustering merger Jaccard threshold (default 0.5; 0 disables)")
    ap.add_argument("--no-api", action="store_true",
                    help="Skip the API call; emit fallback (all GENE_LEVEL_ONLY)")

    args = ap.parse_args(list(argv) if argv is not None else None)

    file_mode = bool(args.gsea_csv or args.esea_csv or args.background_txt)
    if file_mode:
        missing = [name for name, val in (
            ("--gsea-csv", args.gsea_csv),
            ("--esea-csv", args.esea_csv),
            ("--background-txt", args.background_txt),
        ) if val is None]
        if missing:
            ap.error(f"file-path mode requires {', '.join(missing)}")
        if args.dataset:
            ap.error("Pass either --dataset OR file paths, not both.")
    elif not args.dataset:
        ap.error("Pass --dataset NAME or --gsea-csv + --esea-csv + --background-txt.")

    config = AgentConfig(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        gsea_q_threshold=args.q_threshold,
        merge_jaccard_threshold=args.merge_jaccard,
    )

    if file_mode:
        out = run_from_files(
            gsea_csv=args.gsea_csv,
            esea_csv=args.esea_csv,
            background_txt=args.background_txt,
            config=config,
            output_dir=args.output_dir,
            use_api=not args.no_api,
        )
        counts = {"SUPPORTED": 0, "PARTIAL": 0, "GENE_LEVEL_ONLY": 0}
        for v in out.verdicts:
            counts[v.verdict] += 1
        print(f"themes: {len(out.verdicts)}  counts: {counts}")
        return 0

    targets = DATASET_NAMES if args.dataset == "all" else (args.dataset,)
    for d in targets:
        print(f"\n=== Dataset: {d} ===")
        try:
            out = run_dataset(
                d, config,
                inputs_dir=args.inputs_dir,
                outputs_dir=args.output_dir,
                use_api=not args.no_api,
            )
        except APICallError as e:
            print(f"  API error: {e}")
            return 1
        counts = {"SUPPORTED": 0, "PARTIAL": 0, "GENE_LEVEL_ONLY": 0}
        for v in out.verdicts:
            counts[v.verdict] += 1
        print(f"  themes: {len(out.verdicts)}  counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
