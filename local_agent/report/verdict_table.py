"""Generate a markdown verdict table for the report appendix."""
from __future__ import annotations

from pathlib import Path
from typing import List


_VERDICT_PRIORITY = {"SUPPORTED": 0, "PARTIAL": 1, "GENE_LEVEL_ONLY": 2}


def _truncate(text: str, max_len: int) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def render_verdict_table_markdown(
    verdicts: List[dict],
    themes: List[dict],
    rationale_max_chars: int = 180,
) -> str:
    themes_by_id = {t["theme_id"]: t for t in themes}

    rows = sorted(
        verdicts,
        key=lambda v: (
            _VERDICT_PRIORITY[v["verdict"]],
            -abs(themes_by_id.get(v["theme_id"], {}).get("mean_nes", 0.0)),
        ),
    )

    lines = []
    lines.append("| Theme | Dir | NES | Verdict | Weight | Helpers | Rationale |")
    lines.append("|---|---|---|---|---|---|---|")
    for v in rows:
        t = themes_by_id.get(v["theme_id"], {})
        direction = t.get("direction", "")
        nes = t.get("mean_nes")
        nes_str = f"{nes:+.2f}" if nes is not None else ""
        # v2.1: linked_helpers is a list. v2.0 had linked_helper string.
        link_list = v.get("linked_helpers") or (
            [v["linked_helper"]] if v.get("linked_helper") else []
        )
        helpers = ", ".join(link_list) if link_list else "-"
        weight = v.get("theme_weight")
        weight_str = f"{weight:.2f}" if isinstance(weight, (int, float)) else "-"
        rat = _truncate(v.get("rationale", ""), rationale_max_chars)
        label = _truncate(v.get("theme_label", v["theme_id"]), 60).replace("|", "\\|")
        helpers = helpers.replace("|", "\\|")
        rat = rat.replace("|", "\\|")
        lines.append(
            f"| {label} | {direction} | {nes_str} | {v['verdict']} | {weight_str} | {helpers} | {rat} |"
        )
    return "\n".join(lines) + "\n"


def write_verdict_table(
    verdicts: List[dict],
    themes: List[dict],
    output_path: Path,
    rationale_max_chars: int = 180,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    md = render_verdict_table_markdown(verdicts, themes, rationale_max_chars)
    output_path.write_text(md)
    return output_path
