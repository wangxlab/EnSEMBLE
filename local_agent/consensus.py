"""3-run consensus verdict computation (v2.1, multi-helper).

Given 3 reproducibility runs per dataset, compute a deterministic per-theme
consensus verdict by these rules:

  Verdict consensus:
    R1 - 3/3 same verdict     -> ship that verdict
    R2 - 2/3 majority verdict -> ship the majority
    R3 - 3-way split          -> GENE_LEVEL_ONLY (conservative)

  Helper consensus (only for SUPPORTED/PARTIAL):
    Take the UNION of linked_helpers across the runs that voted for the
    consensus verdict, then keep only helpers that appear in >= 2 runs
    (intersection-by-frequency). This filters helper rotation while
    preserving stable links.

  If consensus verdict is SUPPORTED/PARTIAL but the helper intersection is
  empty, fall back to GENE_LEVEL_ONLY (no stable evidence).

Rationale source:
  - R1: pick a run that produced the consensus verdict (lowest index).
  - R2: pick a majority-vote run.
  - R3: synthetic "no consensus" rationale.
  - Helper-empty fallback: synthetic rationale.

Usage:
  python -m local_agent.consensus --output outputs/v2_1
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .config import OUTPUTS_DIR
from .runner import DATASET_NAMES


REPO_DIR = OUTPUTS_DIR / "reproducibility"
DEFAULT_LOCK_DIR = OUTPUTS_DIR / "v2_1"


_GLO_CONSENSUS_RATIONALE = (
    "Consensus across 3 reproducibility runs: verdict was not stable, so this "
    "theme is conservatively classified as GENE_LEVEL_ONLY."
)
_GLO_HELPER_DROPOUT_RATIONALE = (
    "Consensus across 3 reproducibility runs: verdict tier was stable but no "
    "linked helper appeared in >= 2 runs, so no stable regulatory evidence "
    "remains. Conservatively classified as GENE_LEVEL_ONLY."
)


def consensus_verdict_and_helpers(
    runs_for_theme: List[dict],
) -> Tuple[str, List[str], int, List[int]]:
    """Compute (verdict, linked_helpers, rule_id, winning_run_indices) for v2.1.

    runs_for_theme is a list of 3 verdict dicts (one per run) for the same theme_id.

    Rule IDs:
      1 = unanimous verdict
      2 = 2/3 majority verdict
      3 = 3-way split -> GENE_LEVEL_ONLY (no consensus)
    """
    verdicts = [r["verdict"] for r in runs_for_theme]
    counter = Counter(verdicts)
    top_v, top_count = counter.most_common(1)[0]

    if top_count == 3:
        rule = 1
    elif top_count == 2:
        rule = 2
    else:
        # 3-way split
        return "GENE_LEVEL_ONLY", [], 3, []

    winning_indices = [i for i, v in enumerate(verdicts) if v == top_v]

    if top_v == "GENE_LEVEL_ONLY":
        return "GENE_LEVEL_ONLY", [], rule, winning_indices

    # Helper consensus: take union across winning runs, keep helpers in >=2 runs
    helper_count: Counter = Counter()
    for i in winning_indices:
        for h in runs_for_theme[i].get("linked_helpers", []) or []:
            helper_count[h] += 1
    consensus_helpers = sorted([h for h, c in helper_count.items() if c >= 2])

    if not consensus_helpers:
        # No helper survived intersection -> conservative downgrade
        return "GENE_LEVEL_ONLY", [], rule, winning_indices

    return top_v, consensus_helpers, rule, winning_indices


def _pick_rationale(
    runs: List[dict],
    theme_id: str,
    consensus_v: str,
    consensus_helpers: List[str],
    rule_id: int,
    winning_indices: List[int],
) -> str:
    """Pick a rationale text from one of the runs that matches the consensus."""
    if rule_id == 3:
        return _GLO_CONSENSUS_RATIONALE
    if consensus_v == "GENE_LEVEL_ONLY" and not consensus_helpers and winning_indices:
        # Helper intersection was empty
        # (only happens when some runs voted SUPPORTED/PARTIAL but helpers didn't agree)
        # Use synthetic rationale.
        any_winner_was_linked = any(
            runs[i][theme_id]["verdict"] in ("SUPPORTED", "PARTIAL")
            for i in winning_indices
        )
        if any_winner_was_linked:
            return _GLO_HELPER_DROPOUT_RATIONALE
        # Otherwise: GLO was the winning verdict; pick its rationale.
        return runs[winning_indices[0]][theme_id]["rationale"]

    # Find a winning run whose linked_helpers superset matches the consensus set
    consensus_set = set(consensus_helpers)
    best_score = -1
    best_idx = winning_indices[0]
    for i in winning_indices:
        run_helpers = set(runs[i][theme_id].get("linked_helpers", []) or [])
        score = len(run_helpers & consensus_set)
        if score > best_score:
            best_score = score
            best_idx = i
    return runs[best_idx][theme_id]["rationale"]


def compute_consensus_for_dataset(
    dataset: str, repro_dir: Path = REPO_DIR
) -> Tuple[dict, dict]:
    """Return (consensus_output_json, consensus_summary_dict)."""
    runs: List[dict] = []
    for i in (1, 2, 3):
        path = repro_dir / f"run_{i}" / dataset / "verdicts.json"
        runs.append({v["theme_id"]: v for v in json.loads(path.read_text())["verdicts"]})

    theme_ids = sorted(set().union(*[set(r.keys()) for r in runs]))

    consensus_verdicts: List[dict] = []
    rule_counts: Counter = Counter()
    by_rule: dict[int, list[str]] = {1: [], 2: [], 3: []}

    for tid in theme_ids:
        runs_for_theme = [runs[i][tid] for i in range(3)]
        v, helpers, rule, winning = consensus_verdict_and_helpers(runs_for_theme)
        rule_counts[rule] += 1
        by_rule[rule].append(tid)

        label = runs[0][tid]["theme_label"]
        rationale = _pick_rationale(runs, tid, v, helpers, rule, winning)

        consensus_verdicts.append(
            {
                "theme_id": tid,
                "theme_label": label,
                "verdict": v,
                "linked_helpers": helpers,
                "rationale": rationale,
                "consensus_rule": rule,
            }
        )

    out = {"dataset_id": dataset, "verdicts": consensus_verdicts}

    summary = {
        "dataset": dataset,
        "n_themes": len(theme_ids),
        "rule_counts": {f"rule_{k}": rule_counts[k] for k in (1, 2, 3)},
        "by_rule": by_rule,
    }
    return out, summary


def render_consensus_markdown(summaries: List[dict], outputs: List[dict]) -> str:
    rule_descriptions = {
        1: "Unanimous verdict across all 3 runs",
        2: "2/3 majority verdict",
        3: "3-way split -> GENE_LEVEL_ONLY (no consensus)",
    }
    lines = ["# v2.1 Consensus Report (verdict majority + helper intersection)", ""]
    lines.append("Per-dataset consensus rule frequency:")
    lines.append("")
    lines.append("| Dataset | Themes | R1 (unanimous) | R2 (2/3 majority) | R3 (3-way split) |")
    lines.append("|---|---|---|---|---|")
    for s in summaries:
        rc = s["rule_counts"]
        lines.append(
            f"| {s['dataset']} | {s['n_themes']} | {rc['rule_1']} | {rc['rule_2']} | {rc['rule_3']} |"
        )
    lines.append("")
    lines.append("Rule definitions:")
    for k, v in rule_descriptions.items():
        lines.append(f"- **R{k}**: {v}")
    lines.append("")
    lines.append(
        "Helper consensus is computed by union across winning runs, then "
        "keeping helpers that appear in >= 2 runs."
    )
    lines.append("")

    for s, out in zip(summaries, outputs):
        verdicts_by_id = {v["theme_id"]: v for v in out["verdicts"]}
        non_unanimous: list = []
        for rule in (2, 3):
            for tid in s["by_rule"][rule]:
                non_unanimous.append((rule, tid, verdicts_by_id[tid]))
        if not non_unanimous:
            continue
        lines.append(f"## {s['dataset']} - non-unanimous consensus decisions")
        lines.append("")
        lines.append("| Rule | Theme | Verdict | Helpers |")
        lines.append("|---|---|---|---|")
        for rule, tid, v in non_unanimous:
            label = v["theme_label"][:40]
            helpers = ", ".join(v.get("linked_helpers") or []) or "-"
            lines.append(f"| R{rule} | {label} | {v['verdict']} | {helpers} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def write_lockfile(lock_dir: Path) -> None:
    from .config import AgentConfig

    cfg = AgentConfig()
    content = f"""# EnSEMBLE Agent v2.1 — Lockfile

This is the locked configuration for the v2.1 release (multi-helper). The
verdicts in this directory are derived from a 3-run consensus at temperature=0.

## Configuration

```
model:                {cfg.model}
temperature:          {cfg.temperature}
max_tokens:           {cfg.max_tokens}
max_retries:          {cfg.max_retries}
gsea_q_threshold:     {cfg.gsea_q_threshold}
theme_cap_per_dir:    {cfg.theme_cap_per_direction} (0 = unlimited)
theme_cap_total:      {cfg.theme_cap_total}
capacity_cap:         5 themes per helper (raised from 3 in v2.0)
soft_cap_correction:  enabled (removes excess helper from theme; demotes
                      to GENE_LEVEL_ONLY only if no helpers remain)
multi_helper_support: enabled (linked_helpers is a list)
theme_weight:         Stouffer-like sum(-log10 q) / sqrt(n), 0.5x for PARTIAL
clustering:           two-bin auto (locked upstream)
                        N <  250: deepSplit=3, minClusterSize=5
                        N >= 250: deepSplit=1, minClusterSize=10
```

## Consensus policy (v2.1)

For each theme, 3 reproducibility runs at temperature=0 are reconciled in
two stages:

  Verdict consensus:
    R1 - 3/3 runs agree    -> ship that verdict
    R2 - 2/3 majority      -> ship the majority
    R3 - 3-way split       -> GENE_LEVEL_ONLY (conservative)

  Helper consensus (only for SUPPORTED/PARTIAL):
    1. Take the UNION of linked_helpers across runs that voted for the
       consensus verdict.
    2. Keep only helpers appearing in >= 2 of those runs.
    3. If no helper survives, fall back to GENE_LEVEL_ONLY.

The theme_weight stored on each verdict is recomputed against the consensus
helpers, so reported weights reflect the stable evidence only.

## Layout

```
v2_1/
├── lockfile.md                   (this file)
├── consensus_report.md           (per-dataset rule breakdown)
├── reproducibility_report.md     (3-run stability statistics)
├── bt20/
│   ├── verdicts.json             (consensus verdicts; ships)
│   ├── agent_input.json          (input that was sent to the model)
│   ├── api_log_run1.json         (first reproducibility run's full log)
│   └── evaluation.md
├── snai1/  (same layout)
├── panc1/  (same layout)
└── ipsc/   (same layout)
```
"""
    (lock_dir / "lockfile.md").write_text(content)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--repro-dir",
        type=Path,
        default=REPO_DIR,
        help=f"Where the 3-run outputs live (default {REPO_DIR})",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_LOCK_DIR,
        help=f"Where to write the v2.0 lock (default {DEFAULT_LOCK_DIR})",
    )
    ap.add_argument("--datasets", default=",".join(DATASET_NAMES))
    args = ap.parse_args(list(argv) if argv is not None else None)

    targets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    args.output.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    outputs: list[dict] = []

    # Weight recomputation: needs the agent_input.json for each dataset
    # (we use run_1's, since the input is identical across reproducibility runs).
    from .schemas import AgentInput, AgentOutput
    from .weight import annotate_weights

    for dataset in targets:
        out, summary = compute_consensus_for_dataset(dataset, args.repro_dir)

        # Recompute theme_weight against consensus helpers
        run1_input = args.repro_dir / "run_1" / dataset / "agent_input.json"
        if run1_input.exists():
            agent_input = AgentInput.model_validate(json.loads(run1_input.read_text()))
            agent_output = AgentOutput.model_validate(out)
            annotate_weights(agent_input, agent_output)
            out = agent_output.model_dump(mode="json")

        summaries.append(summary)
        outputs.append(out)

        ds_dir = args.output / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / "verdicts.json").write_text(json.dumps(out, indent=2))

        # Copy supplementary artifacts from run_1
        run1 = args.repro_dir / "run_1" / dataset
        for fn in ("agent_input.json",):
            src = run1 / fn
            if src.exists():
                shutil.copy2(src, ds_dir / fn)
        if (run1 / "api_log.json").exists():
            shutil.copy2(run1 / "api_log.json", ds_dir / "api_log_run1.json")

    md = render_consensus_markdown(summaries, outputs)
    (args.output / "consensus_report.md").write_text(md)

    repro_md = args.repro_dir / "reproducibility_report.md"
    if repro_md.exists():
        shutil.copy2(repro_md, args.output / "reproducibility_report.md")

    write_lockfile(args.output)

    # Print summary
    print(f"\n=== v2.1 lock written to {args.output} ===\n")
    for s in summaries:
        rc = s["rule_counts"]
        print(
            f"  {s['dataset']:6s}: themes={s['n_themes']:3d}  "
            f"R1={rc['rule_1']}  R2={rc['rule_2']}  R3={rc['rule_3']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
