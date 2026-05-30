"""Backward-compatible CLI shim.

Accepts both v1.x (Gemini-era) flag spelling and the v2.0 canonical flags.
v1.x flags whose meaning has changed are honored with a deprecation warning
and mapped to the new pipeline; flags that no longer apply are ignored with
a single notice.

Examples
--------

    # v1.x-style (old workflow continues to work):
    python -m local_agent.cli \\
        --gsea-csv GSEA_results.csv \\
        --esea-csv ESEA_helpers.csv \\
        --background-txt background.txt \\
        --output-dir outputs/run1

    # v2.0 canonical:
    python -m local_agent.cli --dataset bt20

    # Anthropic model override:
    python -m local_agent.cli --dataset bt20 --model claude-opus-4-5

    # See the full migration table:
    python -m local_agent.cli --migration-guide
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Iterable

from .caller import APICallError
from .config import INPUTS_DIR, OUTPUTS_DIR, AgentConfig
from .runner import DATASET_NAMES, run_dataset, run_from_files


MIGRATION_TABLE = """\
v1.x flag                            v2.0 mapping
─────────────────────────────────── ──────────────────────────────────────────
--gsea-csv PATH                     supported (file-path mode)
--esea-csv PATH                     supported (file-path mode)
--background-txt PATH               supported (file-path mode)
--output-dir PATH                   supported
--gemini-model NAME                 IGNORED — use --model claude-sonnet-4-5
                                    or --model claude-opus-4-5
--gemini-api-key KEY                IGNORED — export ANTHROPIC_API_KEY
--critic-gemini-model NAME          IGNORED — no critic stage in v2.0
--disable-critic                    IGNORED — no critic stage in v2.0
--gemini-temperature FLOAT          mapped to --temperature (v2.0 default 0)
--gemini-top-p / --gemini-top-k     IGNORED — Anthropic API doesn't expose top_k
--gemini-max-output-tokens INT      mapped to --max-tokens
--gsea-only                         REMOVED — v2.0 requires ESEA helpers
--gsea-top-n INT                    deferred to clustering theme caps
--gsea-q-threshold FLOAT            mapped to --q-threshold
--esea-q-threshold,
  --esea-effect-threshold,
  --esea-partial-q-threshold,
  --esea-partial-effect-threshold   IGNORED — helpers are pre-filtered upstream
--theme-cap INT                     mapped to --theme-cap-per-direction
--theme-top-pathways,
  --theme-leading-edge-target       IGNORED — defaults from locked clustering
--theme-cap-total INT               mapped to --theme-cap-total
--helper-claims-per-theme,
  --esea-max-per-direction          IGNORED — replaced by per-helper cap (5)
                                    and per-theme cap (3)
--resume-stage mini_thesis          replaced by:
                                    python -m local_agent.report.build_thesis \\
                                        --dataset NAME --output-dir DIR

New v2.0 flags (no v1.x equivalent)
─────────────────────────────────── ──────────────────────────────────────────
--model claude-sonnet-4-5 |         Anthropic Claude model
  claude-opus-4-5
--merge-jaccard FLOAT               post-clustering merger threshold (0.5)
--dataset NAME                      convention-driven (inputs/<NAME>/...)
--inputs-dir PATH                   override default inputs/ root
"""


def _warn(msg: str) -> None:
    print(f"[v2.0 migration] {msg}", file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Use --migration-guide for the full v1.x → v2.0 flag mapping.",
    )
    ap.add_argument("--migration-guide", action="store_true",
                    help="Print the v1.x → v2.0 flag-mapping table and exit.")

    # Input selectors (v1.x-style file paths OR v2.0 dataset name)
    ap.add_argument("--dataset", help="Dataset name (or 'all' for the 4 defaults)")
    ap.add_argument("--gsea-csv", type=Path, help="Path to GSEA results CSV (v1.x style)")
    ap.add_argument("--esea-csv", type=Path, help="Path to ESEA helpers CSV (v1.x style)")
    ap.add_argument("--background-txt", type=Path, help="Path to background.txt (v1.x style)")
    ap.add_argument("--inputs-dir", type=Path, default=INPUTS_DIR)
    ap.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)

    # v2.0 canonical flags
    defaults = AgentConfig()
    ap.add_argument("--model", default=defaults.model,
                    help="Anthropic Claude model (claude-sonnet-4-5 or claude-opus-4-5)")
    ap.add_argument("--temperature", type=float, default=defaults.temperature)
    ap.add_argument("--max-tokens", type=int, default=defaults.max_tokens)
    ap.add_argument("--max-retries", type=int, default=defaults.max_retries)
    ap.add_argument("--q-threshold", type=float, default=defaults.gsea_q_threshold)
    ap.add_argument("--theme-cap-per-direction", type=int, default=defaults.theme_cap_per_direction)
    ap.add_argument("--theme-cap-total", type=int, default=defaults.theme_cap_total)
    ap.add_argument("--merge-jaccard", type=float, default=defaults.merge_jaccard_threshold)
    ap.add_argument("--no-api", action="store_true")

    # v1.x compatibility shims (silently mapped or warned)
    ap.add_argument("--gemini-model", dest="_legacy_gemini_model", default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--gemini-api-key", dest="_legacy_gemini_api_key", default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--critic-gemini-model", dest="_legacy_critic_model", default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--disable-critic", dest="_legacy_disable_critic", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--gemini-temperature", dest="_legacy_temperature", type=float, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--gemini-top-p", dest="_legacy_top_p", type=float, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--gemini-top-k", dest="_legacy_top_k", type=int, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--gemini-max-output-tokens", dest="_legacy_max_tokens", type=int, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--gsea-only", dest="_legacy_gsea_only", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--gsea-top-n", dest="_legacy_top_n", type=int, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--esea-q-threshold", dest="_legacy_esea_q", type=float, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--esea-effect-threshold", dest="_legacy_esea_effect", type=float, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--esea-partial-q-threshold", dest="_legacy_partial_q", type=float, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--esea-partial-effect-threshold", dest="_legacy_partial_effect", type=float, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--theme-cap", dest="_legacy_theme_cap", type=int, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--theme-top-pathways", dest="_legacy_theme_top", type=int, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--theme-leading-edge-target", dest="_legacy_le_target", type=int, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--helper-claims-per-theme", dest="_legacy_claims_per_theme", type=int, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--esea-max-per-direction", dest="_legacy_max_per_dir", type=int, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--resume-stage", dest="_legacy_resume_stage", default=None,
                    help=argparse.SUPPRESS)

    return ap


def _apply_legacy_flags(args) -> AgentConfig:
    """Map any v1.x flags the user passed into v2.0 config + emit warnings."""
    # Model
    if args._legacy_gemini_model:
        _warn(
            f"--gemini-model {args._legacy_gemini_model} is ignored; "
            "v2.0 uses Anthropic Claude. Pass --model claude-sonnet-4-5 "
            "or --model claude-opus-4-5."
        )
    if args._legacy_gemini_api_key:
        _warn(
            "--gemini-api-key is ignored; v2.0 requires ANTHROPIC_API_KEY "
            "to be exported in the environment."
        )
    if args._legacy_critic_model or args._legacy_disable_critic:
        _warn("--critic-gemini-model / --disable-critic are ignored "
              "(no critic stage in v2.0).")
    if args._legacy_top_p is not None or args._legacy_top_k is not None:
        _warn("--gemini-top-p / --gemini-top-k are ignored "
              "(Anthropic API does not expose top_k).")
    if args._legacy_temperature is not None:
        _warn(
            f"--gemini-temperature {args._legacy_temperature} mapped to "
            f"--temperature (v2.0 default 0; verify reproducibility intent)."
        )
        args.temperature = args._legacy_temperature
    if args._legacy_max_tokens is not None:
        _warn(f"--gemini-max-output-tokens mapped to --max-tokens "
              f"({args._legacy_max_tokens}).")
        args.max_tokens = args._legacy_max_tokens
    if args._legacy_gsea_only:
        # Hard removal per v2.0 spec.
        raise SystemExit(
            "[v2.0 migration] --gsea-only mode is removed in v2.0. "
            "The agent requires ESEA helpers to classify themes."
        )
    if args._legacy_top_n is not None:
        _warn(f"--gsea-top-n {args._legacy_top_n}: deferred to clustering "
              "theme caps; pass --theme-cap-total instead.")
    for attr, label in (
        ("_legacy_esea_q", "--esea-q-threshold"),
        ("_legacy_esea_effect", "--esea-effect-threshold"),
        ("_legacy_partial_q", "--esea-partial-q-threshold"),
        ("_legacy_partial_effect", "--esea-partial-effect-threshold"),
        ("_legacy_claims_per_theme", "--helper-claims-per-theme"),
        ("_legacy_max_per_dir", "--esea-max-per-direction"),
        ("_legacy_theme_top", "--theme-top-pathways"),
        ("_legacy_le_target", "--theme-leading-edge-target"),
    ):
        if getattr(args, attr) is not None:
            _warn(f"{label} is ignored in v2.0 (helpers are pre-filtered).")
    if args._legacy_theme_cap is not None:
        _warn(f"--theme-cap {args._legacy_theme_cap} mapped to "
              "--theme-cap-per-direction.")
        args.theme_cap_per_direction = args._legacy_theme_cap
    if args._legacy_resume_stage == "mini_thesis":
        raise SystemExit(
            "[v2.0 migration] --resume-stage mini_thesis is now a separate "
            "entry point:\n  python -m local_agent.report.build_thesis "
            "--dataset NAME --output-dir DIR"
        )

    return AgentConfig(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        gsea_q_threshold=args.q_threshold,
        theme_cap_per_direction=args.theme_cap_per_direction,
        theme_cap_total=args.theme_cap_total,
        merge_jaccard_threshold=args.merge_jaccard,
    )


def main(argv: Iterable[str] | None = None) -> int:
    ap = _build_parser()
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.migration_guide:
        print(MIGRATION_TABLE)
        return 0

    config = _apply_legacy_flags(args)

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
        try:
            out = run_from_files(
                gsea_csv=args.gsea_csv,
                esea_csv=args.esea_csv,
                background_txt=args.background_txt,
                config=config,
                output_dir=args.output_dir,
                use_api=not args.no_api,
            )
        except APICallError as e:
            print(f"API error: {e}", file=sys.stderr)
            return 1
        counts = {"SUPPORTED": 0, "PARTIAL": 0, "GENE_LEVEL_ONLY": 0}
        for v in out.verdicts:
            counts[v.verdict] += 1
        print(f"themes: {len(out.verdicts)}  counts: {counts}")
        return 0

    if not args.dataset:
        ap.error("Pass --dataset NAME or v1.x file paths.")

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
            print(f"  API error: {e}", file=sys.stderr)
            return 1
        counts = {"SUPPORTED": 0, "PARTIAL": 0, "GENE_LEVEL_ONLY": 0}
        for v in out.verdicts:
            counts[v.verdict] += 1
        print(f"  themes: {len(out.verdicts)}  counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
