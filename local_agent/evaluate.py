"""Score agent verdicts against ground_truth_v2.md + v2.2 prompt patch criteria.

Each dataset has a list of checks. A check inspects the verdicts and either
passes or fails (or returns N/A if the relevant theme isn't present).

v2.2 evaluator: identity-vs-characteristic distinction.
  _check_supported_not_partial — identity match, must be SUPPORTED
  _check_partial_not_supported — characteristic match, PARTIAL or GLO (NOT SUPPORTED)
  _check_partial               — must be PARTIAL
  _check_count_range           — verdict count must fall in a [min, max] window
  _check_weight_rank           — theme_weight must be in top N among SUPPORTED
  _check_min_helpers           — at least N helpers in linked_helpers
  _check_no_rationale_mention  — rationale (of given verdict tier) must not mention term
  _check_partial_gte_supported — PARTIAL count >= SUPPORTED count for a dataset
  checks_cross_dataset         — global structural checks across all 4 datasets

Usage:
    python -m local_agent.evaluate --dataset bt20
    python -m local_agent.evaluate --dataset all
    python -m local_agent.evaluate --dataset all --output-dir outputs/v2_2_ab/ipsc_sonnet
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from .config import OUTPUTS_DIR
from .schemas import AgentOutput


@dataclass
class CheckResult:
    name: str
    status: str  # "PASS", "FAIL", "N/A"
    detail: str = ""


# Module-level state so check functions don't need to thread the output_dir.
_CURRENT_OUTPUT_DIR: Path = OUTPUTS_DIR


def _set_output_dir(p: Path) -> None:
    global _CURRENT_OUTPUT_DIR
    _CURRENT_OUTPUT_DIR = p


def _load_input_pathways(out: AgentOutput) -> Dict[str, List[str]]:
    inp_path = _CURRENT_OUTPUT_DIR / out.dataset_id / "agent_input.json"
    if not inp_path.exists():
        return {}
    try:
        data = json.loads(inp_path.read_text())
    except Exception:
        return {}
    return {t["theme_id"]: t.get("top_pathways", []) for t in data.get("themes", [])}


def _theme_matches(verdict, needle: str, pathways: List[str]) -> bool:
    """Match a needle against verdict label/id/top_pathways.

    Supports '|' as alternation: 'a|b' matches if either 'a' OR 'b' matches.
    """
    parts = [p.strip().lower() for p in needle.split("|") if p.strip()]
    label = verdict.theme_label.lower()
    tid = verdict.theme_id.lower()
    pathway_blob = " ".join(p.lower() for p in pathways)
    for p in parts:
        if p in label or p in tid or p in pathway_blob:
            return True
    return False


def _all_themes_matching(out: AgentOutput, needle: str) -> List[dict]:
    pathways_by_id = _load_input_pathways(out)
    return [
        v.model_dump()
        for v in out.verdicts
        if _theme_matches(v, needle, pathways_by_id.get(v.theme_id, []))
    ]


def _verdict_counts(out: AgentOutput) -> Dict[str, int]:
    c = {"SUPPORTED": 0, "PARTIAL": 0, "GENE_LEVEL_ONLY": 0}
    for v in out.verdicts:
        c[v.verdict] += 1
    return c


def _helpers_str(m: dict) -> str:
    helpers = m.get("linked_helpers") or []
    return ", ".join(helpers) if helpers else "-"


# ---------- Original primitives (kept for compatibility) ----------


def _check_supported(out: AgentOutput, label_substr: str, helper_substr: Optional[str] = None) -> CheckResult:
    name = f"'{label_substr}' is SUPPORTED"
    if helper_substr:
        name += f" with helper containing '{helper_substr}'"
    matches = _all_themes_matching(out, label_substr)
    if not matches:
        return CheckResult(name, "N/A", f"no theme matched '{label_substr}'")
    for m in matches:
        if m["verdict"] == "SUPPORTED":
            helpers = m.get("linked_helpers") or []
            if helper_substr is None:
                return CheckResult(name, "PASS", f"{m['theme_id']} -> SUPPORTED ({_helpers_str(m)})")
            need_parts = [p.strip().upper() for p in helper_substr.split("|")]
            if any(any(p in h.upper() for p in need_parts) for h in helpers):
                return CheckResult(name, "PASS", f"{m['theme_id']} -> SUPPORTED ({_helpers_str(m)})")
    return CheckResult(name, "FAIL", "; ".join(f"{m['theme_id']}={m['verdict']} (helpers={_helpers_str(m)})" for m in matches))


def _check_glo(out: AgentOutput, label_substr: str) -> CheckResult:
    name = f"'{label_substr}' is GENE_LEVEL_ONLY"
    matches = _all_themes_matching(out, label_substr)
    if not matches:
        return CheckResult(name, "N/A", f"no theme matched '{label_substr}'")
    bad = [m for m in matches if m["verdict"] != "GENE_LEVEL_ONLY"]
    if not bad:
        return CheckResult(name, "PASS", f"{len(matches)} theme(s) all GLO")
    return CheckResult(name, "FAIL", "; ".join(f"{m['theme_id']}={m['verdict']}" for m in bad))


def _check_not_supported(out: AgentOutput, label_substr: str) -> CheckResult:
    name = f"'{label_substr}' is NOT SUPPORTED"
    matches = _all_themes_matching(out, label_substr)
    if not matches:
        return CheckResult(name, "N/A", f"no theme matched '{label_substr}'")
    bad = [m for m in matches if m["verdict"] == "SUPPORTED"]
    if not bad:
        return CheckResult(name, "PASS", "; ".join(f"{m['theme_id']}={m['verdict']}" for m in matches))
    return CheckResult(name, "FAIL", "; ".join(f"{m['theme_id']}=SUPPORTED ({_helpers_str(m)})" for m in bad))


# ---------- New v2.2 primitives ----------


def _check_partial(out: AgentOutput, label_substr: str) -> CheckResult:
    """At least one matching theme must be PARTIAL."""
    name = f"'{label_substr}' is PARTIAL"
    matches = _all_themes_matching(out, label_substr)
    if not matches:
        return CheckResult(name, "N/A", f"no theme matched '{label_substr}'")
    if any(m["verdict"] == "PARTIAL" for m in matches):
        return CheckResult(name, "PASS", "; ".join(f"{m['theme_id']}={m['verdict']}" for m in matches if m["verdict"] == "PARTIAL"))
    return CheckResult(name, "FAIL", "; ".join(f"{m['theme_id']}={m['verdict']}" for m in matches))


def _check_supported_not_partial(out: AgentOutput, label_substr: str, helper_substr: Optional[str] = None) -> CheckResult:
    """At least one matching theme must be SUPPORTED (PARTIAL not enough). Identity match."""
    name = f"'{label_substr}' is SUPPORTED (identity match)"
    if helper_substr:
        name += f" with helper '{helper_substr}'"
    matches = _all_themes_matching(out, label_substr)
    if not matches:
        return CheckResult(name, "N/A", f"no theme matched '{label_substr}'")
    supported = [m for m in matches if m["verdict"] == "SUPPORTED"]
    if not supported:
        return CheckResult(name, "FAIL", "; ".join(f"{m['theme_id']}={m['verdict']}" for m in matches))
    if helper_substr:
        need_parts = [p.strip().upper() for p in helper_substr.split("|")]
        ok = [m for m in supported if any(any(p in h.upper() for p in need_parts) for h in (m.get("linked_helpers") or []))]
        if not ok:
            return CheckResult(name, "FAIL", "SUPPORTED but no matching helper: " + "; ".join(f"{m['theme_id']} helpers={_helpers_str(m)}" for m in supported))
        return CheckResult(name, "PASS", f"{ok[0]['theme_id']} -> SUPPORTED ({_helpers_str(ok[0])})")
    return CheckResult(name, "PASS", f"{supported[0]['theme_id']} -> SUPPORTED ({_helpers_str(supported[0])})")


def _check_partial_not_supported(out: AgentOutput, label_substr: str) -> CheckResult:
    """Matching theme must be PARTIAL or GLO, NOT SUPPORTED. Characteristic match."""
    name = f"'{label_substr}' is PARTIAL or GLO (characteristic, not identity)"
    matches = _all_themes_matching(out, label_substr)
    if not matches:
        return CheckResult(name, "N/A", f"no theme matched '{label_substr}'")
    bad = [m for m in matches if m["verdict"] == "SUPPORTED"]
    if bad:
        return CheckResult(name, "FAIL", "; ".join(f"{m['theme_id']}=SUPPORTED ({_helpers_str(m)})" for m in bad))
    return CheckResult(name, "PASS", "; ".join(f"{m['theme_id']}={m['verdict']}" for m in matches))


def _check_count_range(out: AgentOutput, verdict: str, min_count: int = 0, max_count: int = 999) -> CheckResult:
    name = f"{verdict} count in [{min_count}, {max_count}]"
    n = sum(1 for v in out.verdicts if v.verdict == verdict)
    if min_count <= n <= max_count:
        return CheckResult(name, "PASS", f"{verdict}={n}")
    return CheckResult(name, "FAIL", f"{verdict}={n}")


def _check_weight_rank(out: AgentOutput, label_substr: str, max_rank: int = 3) -> CheckResult:
    name = f"'{label_substr}' weight in top {max_rank} of SUPPORTED themes"
    matches = [v for v in out.verdicts if _theme_matches(v, label_substr, _load_input_pathways(out).get(v.theme_id, [])) and v.verdict == "SUPPORTED"]
    if not matches:
        return CheckResult(name, "N/A", f"no SUPPORTED theme matched '{label_substr}'")
    all_supported = sorted(
        [v for v in out.verdicts if v.verdict == "SUPPORTED"],
        key=lambda v: -(v.theme_weight or 0),
    )
    target_w = max((m.theme_weight or 0) for m in matches)
    rank = next((i + 1 for i, v in enumerate(all_supported) if (v.theme_weight or 0) == target_w), len(all_supported))
    if rank <= max_rank:
        return CheckResult(name, "PASS", f"rank={rank}, weight={target_w:.2f}")
    return CheckResult(name, "FAIL", f"rank={rank}, weight={target_w:.2f}")


def _check_min_helpers(out: AgentOutput, label_substr: str, min_helpers: int = 2) -> CheckResult:
    name = f"'{label_substr}' has >= {min_helpers} linked helpers"
    matches = _all_themes_matching(out, label_substr)
    sp = [m for m in matches if m["verdict"] in ("SUPPORTED", "PARTIAL")]
    if not sp:
        return CheckResult(name, "N/A", f"no SUPPORTED/PARTIAL theme matched '{label_substr}'")
    best = max(sp, key=lambda m: len(m.get("linked_helpers") or []))
    n = len(best.get("linked_helpers") or [])
    if n >= min_helpers:
        return CheckResult(name, "PASS", f"{best['theme_id']} has {n} helpers ({_helpers_str(best)})")
    return CheckResult(name, "FAIL", f"{best['theme_id']} has only {n} helper(s)")


def _check_no_rationale_mention(out: AgentOutput, term: str, verdict: str = "SUPPORTED") -> CheckResult:
    name = f"No {verdict} rationale mentions '{term}'"
    bad = [v.model_dump() for v in out.verdicts if v.verdict == verdict and term.lower() in (v.rationale or "").lower()]
    if not bad:
        return CheckResult(name, "PASS", f"0 {verdict} rationales mention '{term}'")
    return CheckResult(name, "FAIL", "; ".join(f"{m['theme_id']}" for m in bad))


def _check_partial_gte_supported(out: AgentOutput) -> CheckResult:
    name = "PARTIAL count >= SUPPORTED count (more characteristic than identity matches)"
    n_sup = sum(1 for v in out.verdicts if v.verdict == "SUPPORTED")
    n_par = sum(1 for v in out.verdicts if v.verdict == "PARTIAL")
    if n_par >= n_sup:
        return CheckResult(name, "PASS", f"PARTIAL={n_par}, SUPPORTED={n_sup}")
    return CheckResult(name, "FAIL", f"PARTIAL={n_par}, SUPPORTED={n_sup}")


# --- Per-dataset checklists (v2.2) -------------------------------------------


def checks_bt20(out: AgentOutput) -> List[CheckResult]:
    return [
        # Identity match
        _check_supported_not_partial(out, "epithelial-mesenchymal", "MESENCHYMAL"),
        # Must be GLO (no functional connection to any helper)
        _check_glo(out, "e2f"),
        _check_glo(out, "dna-repair"),
        _check_glo(out, "g2-m"),
        _check_glo(out, "sister-chromatid"),
        _check_glo(out, "notch"),
        # Artifacts
        _check_not_supported(out, "interferon"),
        # Structural
        _check_count_range(out, "SUPPORTED", min_count=1, max_count=3),
        _check_count_range(out, "PARTIAL", min_count=0, max_count=3),
        _check_count_range(out, "GENE_LEVEL_ONLY", min_count=28, max_count=35),
        # No Wnt hallucination
        _check_no_rationale_mention(out, "wnt", verdict="SUPPORTED"),
    ]


def checks_snai1(out: AgentOutput) -> List[CheckResult]:
    return [
        # Identity matches -> SUPPORTED
        _check_supported_not_partial(out, "epithelial-mesenchymal", "FIBROBLAST|HF1|MYOTUBE|HME|IMR"),
        _check_supported_not_partial(out, "tissue-migration", "FIBROBLAST|HF1|MYOTUBE"),
        # EMT must have multiple helpers (identity match attracts strong fibroblast helpers)
        _check_min_helpers(out, "epithelial-mesenchymal", min_helpers=2),
        # EMT weight must be in top 3
        _check_weight_rank(out, "epithelial-mesenchymal", max_rank=3),
        # Characteristic matches -> PARTIAL (NOT SUPPORTED)
        _check_partial_not_supported(out, "tnf"),
        _check_partial_not_supported(out, "inflammatory"),
        # Structural
        _check_count_range(out, "SUPPORTED", min_count=2, max_count=5),
        _check_count_range(out, "PARTIAL", min_count=2, max_count=8),
        # No biologically implausible links (UP themes have no UP helpers in SNAI1)
        _check_glo(out, "ribosome"),
        _check_glo(out, "trna"),
    ]


def checks_panc1(out: AgentOutput) -> List[CheckResult]:
    return [
        # Identity matches via hallmark -> SUPPORTED
        _check_supported_not_partial(out, "myc-targets", "SUPT5H"),
        _check_supported_not_partial(out, "mtorc1"),
        _check_supported_not_partial(out, "slits-and-robos|translation"),
        # UP neural themes must stay GLO (artifact in PANC-1 at 1h)
        _check_glo(out, "synapse"),
        _check_glo(out, "neurogenesis"),
        _check_glo(out, "ion-transmembrane"),
        _check_not_supported(out, "epithelial-mesenchymal"),  # 1h too fast for EMT
        # Structural
        _check_count_range(out, "SUPPORTED", min_count=2, max_count=5),
        _check_count_range(out, "PARTIAL", min_count=0, max_count=5),
    ]


def checks_ipsc(out: AgentOutput) -> List[CheckResult]:
    return [
        # UP identity matches -> SUPPORTED with neural helpers
        _check_supported_not_partial(out, "signal-release|synaptic", "GLUTAMATERGIC|GABA|NEURON"),
        _check_supported_not_partial(out, "junction-assembly|synap", "GLUTAMATERGIC|NEURON"),
        _check_supported_not_partial(out, "projection-morphogenesis|axon"),
        _check_supported_not_partial(out, "nervous-system|neurogenesis"),
        # UP artifacts -> GLO
        _check_glo(out, "cilium"),
        _check_glo(out, "flagellum"),
        # DOWN identity match -> SUPPORTED
        _check_supported_not_partial(out, "transcription-elongation", "POLR2A"),
        # DOWN characteristic matches -> PARTIAL (NOT SUPPORTED)
        _check_partial_not_supported(out, "e2f"),
        _check_partial_not_supported(out, "rna-processing"),
        _check_partial_not_supported(out, "mitotic-cell-cycle|negative-regulation-of-mitotic"),
        # DOWN artifacts -> GLO
        _check_glo(out, "influenza"),
        _check_glo(out, "ifn"),
        # Structural: PARTIAL must exist and be substantial
        _check_count_range(out, "SUPPORTED", min_count=5, max_count=15),
        _check_count_range(out, "PARTIAL", min_count=4, max_count=15),
        # iPSC should have more PARTIAL than SUPPORTED (many characteristics)
        _check_partial_gte_supported(out),
    ]


CHECKS: Dict[str, Callable[[AgentOutput], List[CheckResult]]] = {
    "bt20": checks_bt20,
    "snai1": checks_snai1,
    "panc1": checks_panc1,
    "ipsc": checks_ipsc,
}


# --- Cross-dataset structural checks -----------------------------------------


def checks_cross_dataset(all_outputs: Dict[str, AgentOutput]) -> List[CheckResult]:
    results: List[CheckResult] = []

    # PARTIAL must exist in >= 3 of 4 datasets
    datasets_with_partial = sum(
        1
        for out in all_outputs.values()
        if sum(1 for v in out.verdicts if v.verdict == "PARTIAL") >= 2
    )
    n_total = len(all_outputs)
    threshold = max(1, n_total - 1) if n_total >= 2 else 1
    results.append(
        CheckResult(
            f"PARTIAL category alive in >= {threshold}/{n_total} datasets",
            "PASS" if datasets_with_partial >= threshold else "FAIL",
            f"{datasets_with_partial}/{n_total} datasets have >= 2 PARTIAL",
        )
    )

    # BT20 no regression: SUPPORTED count <= 3
    bt20 = all_outputs.get("bt20")
    if bt20:
        n_sup = sum(1 for v in bt20.verdicts if v.verdict == "SUPPORTED")
        results.append(
            CheckResult(
                "BT20 no regression (SUPPORTED <= 3)",
                "PASS" if n_sup <= 3 else "FAIL",
                f"BT20 SUPPORTED={n_sup} (v2.1 had 1)",
            )
        )

    # PANC1 MYC targets still SUPPORTED
    panc1 = all_outputs.get("panc1")
    if panc1:
        pathways = _load_input_pathways(panc1)
        has_myc = any(
            v.verdict == "SUPPORTED" and _theme_matches(v, "myc-targets", pathways.get(v.theme_id, []))
            for v in panc1.verdicts
        )
        results.append(
            CheckResult(
                "PANC1 MYC targets still SUPPORTED",
                "PASS" if has_myc else "FAIL",
                "regression check",
            )
        )

    # SNAI1 EMT weight > Inflammatory weight
    snai1 = all_outputs.get("snai1")
    if snai1:
        pathways = _load_input_pathways(snai1)
        emt_w = max(
            ((v.theme_weight or 0) for v in snai1.verdicts
             if _theme_matches(v, "epithelial-mesenchymal", pathways.get(v.theme_id, []))),
            default=0.0,
        )
        inf_w = max(
            ((v.theme_weight or 0) for v in snai1.verdicts
             if _theme_matches(v, "inflammatory", pathways.get(v.theme_id, []))),
            default=0.0,
        )
        results.append(
            CheckResult(
                "SNAI1 EMT weight > Inflammatory weight",
                "PASS" if emt_w > inf_w else "FAIL",
                f"EMT={emt_w:.2f}, Inflammatory={inf_w:.2f}",
            )
        )

    return results


# --- Entry points ------------------------------------------------------------


def evaluate(dataset: str, outputs_dir: Path = OUTPUTS_DIR):
    _set_output_dir(outputs_dir)
    verdicts_path = outputs_dir / dataset / "verdicts.json"
    if not verdicts_path.exists():
        raise FileNotFoundError(f"No verdicts.json at {verdicts_path}")
    out = AgentOutput.model_validate(json.loads(verdicts_path.read_text()))
    out.dataset_id = dataset
    return CHECKS[dataset](out), out


def render_markdown(dataset: str, results: List[CheckResult], out: AgentOutput) -> str:
    counts = _verdict_counts(out)
    lines = [f"# Evaluation: {dataset}", ""]
    lines.append(f"- Themes: {len(out.verdicts)}")
    lines.append(f"- Counts: {counts}")
    lines.append("")
    lines.append("| Check | Status | Detail |")
    lines.append("|---|---|---|")
    for r in results:
        lines.append(f"| {r.name} | {r.status} | {r.detail} |")
    return "\n".join(lines) + "\n"


def render_cross_markdown(results: List[CheckResult]) -> str:
    lines = ["# Cross-dataset structural checks", "", "| Check | Status | Detail |", "|---|---|---|"]
    for r in results:
        lines.append(f"| {r.name} | {r.status} | {r.detail} |")
    return "\n".join(lines) + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, help="Dataset name or 'all'")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR,
        help="Where to find <dataset>/verdicts.json (default: outputs/)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    targets = list(CHECKS.keys()) if args.dataset == "all" else [args.dataset]
    if args.dataset != "all" and args.dataset not in CHECKS:
        ap.error(f"unknown dataset '{args.dataset}'")

    overall_fail = False
    all_outputs: Dict[str, AgentOutput] = {}
    for d in targets:
        results, out = evaluate(d, outputs_dir=args.output_dir)
        all_outputs[d] = out
        md = render_markdown(d, results, out)
        out_md = args.output_dir / d / "evaluation.md"
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md)
        print(md)
        if any(r.status == "FAIL" for r in results):
            overall_fail = True

    if len(targets) > 1:
        cross = checks_cross_dataset(all_outputs)
        cross_md = render_cross_markdown(cross)
        (args.output_dir / "cross_dataset_evaluation.md").write_text(cross_md)
        print(cross_md)
        if any(r.status == "FAIL" for r in cross):
            overall_fail = True

    return 1 if overall_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
