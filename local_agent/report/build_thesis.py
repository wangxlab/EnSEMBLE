"""Generate mini-thesis markdown for one or more datasets.

Reads agent_input.json + verdicts.json from <output-dir>/<dataset>/, calls
Claude (Sonnet 4.5 default per spec), validates the prose, and writes
mini_thesis.md + thesis_validation.json next to them.

Usage:
    python -m local_agent.report.build_thesis --dataset bt20
    python -m local_agent.report.build_thesis --dataset all --output-dir outputs/v2_2_lock
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from ..config import OUTPUTS_DIR
from ..runner import DATASET_NAMES
from .thesis_caller import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    build_thesis_input,
    generate_mini_thesis,
)
from .thesis_validator import validate_mini_thesis


V22_LOCK_DIR = OUTPUTS_DIR / "v2_2_lock"


def generate_for_dataset(
    dataset: str,
    output_dir: Path,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Path:
    ds_dir = output_dir / dataset
    inp_path = ds_dir / "agent_input.json"
    vp = ds_dir / "verdicts.json"
    if not inp_path.exists() or not vp.exists():
        raise FileNotFoundError(f"Missing inputs at {ds_dir}")

    agent_input = json.loads(inp_path.read_text())
    verdicts = json.loads(vp.read_text())
    thesis_input = build_thesis_input(agent_input, verdicts)

    text, api_log = generate_mini_thesis(
        thesis_input,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    helper_names = [h["helper_name"] for h in agent_input["helpers"]]
    warnings = validate_mini_thesis(text, helper_names)

    out_md = ds_dir / "mini_thesis.md"
    out_md.write_text(text + ("\n" if not text.endswith("\n") else ""))
    (ds_dir / "thesis_api_log.json").write_text(json.dumps(api_log, indent=2, default=str))
    (ds_dir / "thesis_validation.json").write_text(
        json.dumps(
            {
                "warnings": warnings,
                "word_count": len(text.split()),
                "model": model,
                "temperature": temperature,
            },
            indent=2,
        )
    )
    return out_md


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, help="dataset name or 'all'")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=V22_LOCK_DIR,
        help=f"directory containing <dataset>/{{agent_input,verdicts}}.json (default {V22_LOCK_DIR})",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    args = ap.parse_args(list(argv) if argv is not None else None)

    targets = list(DATASET_NAMES) if args.dataset == "all" else [args.dataset]
    for ds in targets:
        print(f"\n=== Generating mini-thesis for {ds} ===")
        path = generate_for_dataset(
            ds,
            output_dir=args.output_dir,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        meta = json.loads((path.parent / "thesis_validation.json").read_text())
        print(f"  -> {path}")
        print(f"  word_count={meta['word_count']}  warnings={len(meta['warnings'])}")
        for w in meta["warnings"]:
            print(f"    WARN: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
