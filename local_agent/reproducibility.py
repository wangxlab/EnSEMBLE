"""Reproducibility check: run each dataset N times at temperature=0
and diff verdicts across runs (v2.1 multi-helper aware).

Pass criteria: zero UNSTABLE themes (every theme has the same verdict
and identical linked_helpers set across all runs).

Usage:
    python -m local_agent.reproducibility --runs 3
    python -m local_agent.reproducibility --runs 3 --datasets bt20,snai1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .config import OUTPUTS_DIR, AgentConfig
from .runner import run_dataset, DATASET_NAMES


def _load_verdicts(path: Path) -> dict:
    return {v["theme_id"]: v for v in json.loads(path.read_text())["verdicts"]}


def diff_runs(dataset: str, run_dirs: list[Path]) -> dict:
    """Compare verdicts across N run directories. Return per-theme stability."""
    runs = [_load_verdicts(d / "verdicts.json") for d in run_dirs]
    theme_ids = set().union(*[set(r.keys()) for r in runs])

    stable: list[str] = []
    unstable: list[dict] = []

    for tid in sorted(theme_ids):
        verdicts = [r[tid]["verdict"] for r in runs]
        helper_sets = [tuple(sorted(r[tid].get("linked_helpers") or [])) for r in runs]
        if len(set(verdicts)) == 1 and len(set(helper_sets)) == 1:
            stable.append(tid)
        else:
            unstable.append(
                {
                    "theme_id": tid,
                    "theme_label": runs[0][tid].get("theme_label"),
                    "verdicts": verdicts,
                    "linked_helpers": [list(s) for s in helper_sets],
                }
            )

    return {
        "dataset": dataset,
        "n_runs": len(runs),
        "n_themes": len(theme_ids),
        "n_stable": len(stable),
        "n_unstable": len(unstable),
        "unstable_themes": unstable,
    }


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    defaults = AgentConfig()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--datasets", default=",".join(DATASET_NAMES))
    ap.add_argument("--model", default=defaults.model)
    ap.add_argument("--max-tokens", type=int, default=defaults.max_tokens)
    ap.add_argument(
        "--merge-jaccard", type=float, default=defaults.merge_jaccard_threshold
    )
    ap.add_argument(
        "--repro-dir",
        type=Path,
        default=OUTPUTS_DIR / "reproducibility",
        help="Where to store per-run outputs",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    targets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    config = AgentConfig(
        model=args.model,
        max_tokens=args.max_tokens,
        merge_jaccard_threshold=args.merge_jaccard,
    )

    args.repro_dir.mkdir(parents=True, exist_ok=True)

    overall_unstable = 0
    reports: list[dict] = []

    for dataset in targets:
        run_dirs: list[Path] = []
        for i in range(1, args.runs + 1):
            out_dir = args.repro_dir / dataset / f"run_{i}"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n=== {dataset} run {i}/{args.runs} ===")
            run_dataset(
                dataset=dataset,
                config=config,
                outputs_dir=args.repro_dir / f"run_{i}",
            )
            # run_dataset wrote into <repro_dir>/run_i/<dataset>/.
            actual_out = args.repro_dir / f"run_{i}" / dataset
            run_dirs.append(actual_out)

        report = diff_runs(dataset, run_dirs)
        reports.append(report)
        overall_unstable += report["n_unstable"]
        print(
            f"\n{dataset}: {report['n_stable']}/{report['n_themes']} stable, "
            f"{report['n_unstable']} unstable"
        )
        if report["unstable_themes"]:
            for u in report["unstable_themes"]:
                print(f"  UNSTABLE: {u['theme_id']}  verdicts={u['verdicts']}  helpers={u['linked_helpers']}")

    out_md = render_markdown(reports, args.runs)
    md_path = args.repro_dir / "reproducibility_report.md"
    md_path.write_text(out_md)
    print(f"\nReport written to: {md_path}")

    return 0 if overall_unstable == 0 else 1


def render_markdown(reports: list[dict], n_runs: int) -> str:
    lines = [f"# Reproducibility Report (N={n_runs} runs at temperature=0)", ""]
    lines.append("| Dataset | Themes | Stable | Unstable | Pass |")
    lines.append("|---|---|---|---|---|")
    for r in reports:
        passed = "✅" if r["n_unstable"] == 0 else "❌"
        lines.append(
            f"| {r['dataset']} | {r['n_themes']} | {r['n_stable']} | {r['n_unstable']} | {passed} |"
        )
    lines.append("")
    for r in reports:
        if r["unstable_themes"]:
            lines.append(f"## Unstable themes — {r['dataset']}")
            lines.append("")
            for u in r["unstable_themes"]:
                lines.append(f"- **{u['theme_id']}** — {u['theme_label']}")
                lines.append(f"  - verdicts across runs: `{u['verdicts']}`")
                lines.append(f"  - linked_helpers: `{u['linked_helpers']}`")
            lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
