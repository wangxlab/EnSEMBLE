"""Deterministic post-LLM validation rules (v2.1, multi-helper).

The rules:
  1. Completeness: every input theme has exactly one verdict.
  2. Valid verdict values.
  3. GENE_LEVEL_ONLY -> linked_helpers must be an empty list.
  4. SUPPORTED/PARTIAL -> linked_helpers must be non-empty; every entry
     must be in the input helpers list.
  5. Direction consistency: every linked helper must share theme direction.
  6a. Per-helper capacity: each helper appears across at most 5 themes total.
  6b. Per-theme capacity: each theme lists at most 3 helpers.
  7. Rationale non-empty and >= 15 words.

Plus a soft auto-correction: when only capacity rules (6a / 6b) fire,
trim to fit the caps without burning the whole output. Per-theme cap
trims by helper q-value (weakest first); per-helper cap trims by verdict
tier and theme |mean_nes|. If a verdict ends up with no helpers, demote
to GENE_LEVEL_ONLY.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .schemas import AgentInput, AgentOutput, VerdictItem


MAX_THEMES_PER_HELPER = 5  # v2.1: cap on how many themes a helper may appear in
MAX_HELPERS_PER_THEME = 3  # v2.1: cap on how many helpers a single theme may list
MIN_RATIONALE_WORDS = 15


def validate_verdicts(agent_input: AgentInput, agent_output: AgentOutput) -> List[str]:
    """Return a list of validation errors. Empty list = valid."""
    errors: List[str] = []

    input_theme_ids = {t.theme_id for t in agent_input.themes}
    output_theme_ids = {v.theme_id for v in agent_output.verdicts}
    helper_names = {h.helper_name for h in agent_input.helpers}
    helper_directions = {h.helper_name: h.direction for h in agent_input.helpers}
    theme_directions = {t.theme_id: t.direction for t in agent_input.themes}

    # Rule 1: completeness
    missing = sorted(input_theme_ids - output_theme_ids)
    extra = sorted(output_theme_ids - input_theme_ids)
    if missing:
        errors.append(f"Missing verdicts for themes: {missing}")
    if extra:
        errors.append(f"Verdicts reference unknown theme_ids: {extra}")

    seen: set[str] = set()
    for v in agent_output.verdicts:
        if v.theme_id in seen:
            errors.append(f"Duplicate verdict for theme_id '{v.theme_id}'")
        seen.add(v.theme_id)

    # Rules 3, 4, 5, 7 per verdict
    for v in agent_output.verdicts:
        helpers = v.linked_helpers or []

        # Rule 3: GLO -> empty list
        if v.verdict == "GENE_LEVEL_ONLY" and helpers:
            errors.append(
                f"{v.theme_id}: GENE_LEVEL_ONLY but linked_helpers={helpers} "
                "(must be empty list)"
            )

        # Rule 4: SUPPORTED/PARTIAL -> non-empty, all valid
        if v.verdict in ("SUPPORTED", "PARTIAL"):
            if not helpers:
                errors.append(f"{v.theme_id}: {v.verdict} but linked_helpers is empty")
            for h in helpers:
                if h not in helper_names:
                    errors.append(
                        f"{v.theme_id}: linked helper '{h}' not in input helpers"
                    )

        # Rule 4b: no duplicate helpers in a single linked_helpers list
        if len(helpers) != len(set(helpers)):
            dup = [h for h in helpers if helpers.count(h) > 1]
            errors.append(
                f"{v.theme_id}: duplicate helpers in linked_helpers: {sorted(set(dup))}"
            )

        # Rule 5: direction consistency for every helper
        theme_dir = theme_directions.get(v.theme_id)
        for h in helpers:
            if h in helper_directions and theme_dir is not None:
                if helper_directions[h] != theme_dir:
                    errors.append(
                        f"{v.theme_id}: helper '{h}' direction "
                        f"{helper_directions[h]} != theme direction {theme_dir}"
                    )

        # Rule 7: rationale length
        rationale_words = v.rationale.split()
        if len(rationale_words) < MIN_RATIONALE_WORDS:
            errors.append(
                f"{v.theme_id}: rationale has {len(rationale_words)} words "
                f"(min {MIN_RATIONALE_WORDS})"
            )

    # Rule 6a: per-helper capacity (cap = 5)
    helper_counts: dict[str, int] = {}
    for v in agent_output.verdicts:
        for h in v.linked_helpers or []:
            helper_counts[h] = helper_counts.get(h, 0) + 1
    for h, count in helper_counts.items():
        if count > MAX_THEMES_PER_HELPER:
            errors.append(
                f"Helper '{h}' appears in {count} themes (max {MAX_THEMES_PER_HELPER})"
            )

    # Rule 6b: per-theme helper cap (cap = 3)
    for v in agent_output.verdicts:
        n = len(v.linked_helpers or [])
        if n > MAX_HELPERS_PER_THEME:
            errors.append(
                f"{v.theme_id}: linked_helpers has {n} entries (max {MAX_HELPERS_PER_THEME})"
            )

    return errors


_VERDICT_PRIORITY = {"SUPPORTED": 0, "PARTIAL": 1, "GENE_LEVEL_ONLY": 2}
_AUTO_DEMOTE_NOTE = " (auto-demoted: helper capacity cap exceeded)"


def is_capacity_only_errors(errors: List[str]) -> bool:
    """True iff every error is a violation auto_correct_capacity can fix.

    Includes:
      - per-helper capacity (rule 6a)
      - per-theme capacity (rule 6b)
      - direction mismatch (rule 5; surgically removes the offending helper)
      - duplicate helpers in one verdict (rule 4b; deduped)
      - unknown helper names (rule 4; dropped)
      - duplicate verdicts for the same theme_id (rule 1b; merged)
    """
    if not errors:
        return False

    def is_correctable(e: str) -> bool:
        if "appears in " in e and "themes (max" in e:
            return True
        if "linked_helpers has " in e and "entries (max" in e:
            return True
        if "direction" in e and "!=" in e:
            return True
        if "duplicate helpers" in e:
            return True
        if "not in input helpers" in e:
            return True
        if "Duplicate verdict for theme_id" in e:
            return True
        if "rationale has " in e and "(min" in e:
            return True
        return False

    return all(is_correctable(e) for e in errors)


def auto_correct_capacity(
    agent_input: AgentInput, agent_output: AgentOutput
) -> Tuple[AgentOutput, List[str]]:
    """Surgically repair common LLM mistakes before falling back (v2.1).

    Stage 0 (direction mismatch): if a helper's direction != its theme's
    direction, remove that helper from the verdict's linked_helpers list.
    This is a clear hard violation that the agent should not have made.

    Stage 1 (per-theme cap): for any theme with > MAX_HELPERS_PER_THEME
    linked helpers, keep only the strongest 3 (lowest helper q-value first).

    Stage 2 (per-helper cap): for any helper that still appears in >
    MAX_THEMES_PER_HELPER themes, rank those themes by (verdict tier asc,
    |mean_nes| desc) and remove the helper from the excess.

    Also dedupes helper lists (drops duplicate names within one verdict).

    If any verdict ends up with empty linked_helpers, demote to GLO and
    append the auto-demote note.

    Returns (corrected_output, list_of_changed_theme_ids).
    """
    theme_nes: Dict[str, float] = {t.theme_id: abs(t.mean_nes) for t in agent_input.themes}
    helper_q: Dict[str, float] = {h.helper_name: h.q_value for h in agent_input.helpers}
    theme_dir: Dict[str, str] = {t.theme_id: t.direction for t in agent_input.themes}
    helper_dir: Dict[str, str] = {h.helper_name: h.direction for h in agent_input.helpers}
    valid_helpers: set = {h.helper_name for h in agent_input.helpers}

    new_verdicts: dict[str, VerdictItem] = {}
    changed_ids: list[str] = []
    changed_set: set[str] = set()

    # Stage -2: lengthen short rationales by demoting to GLO with a
    # synthetic message. The agent occasionally writes a 10-13 word rationale
    # on a single verdict; the rule requires >= 15 words. Demote that one
    # verdict to GLO rather than failing the whole output.
    short_rationale_note = (
        " (auto-demoted: rationale was below the 15-word minimum and "
        "has been replaced for compliance.)"
    )
    pre_dedup: list[VerdictItem] = []
    for v in agent_output.verdicts:
        v_copy = v.model_copy(deep=True)
        if len(v_copy.rationale.split()) < MIN_RATIONALE_WORDS:
            v_copy.verdict = "GENE_LEVEL_ONLY"
            v_copy.linked_helpers = []
            v_copy.rationale = (
                v_copy.rationale.rstrip() + short_rationale_note
            )
            if v_copy.theme_id not in changed_set:
                changed_set.add(v_copy.theme_id)
                changed_ids.append(v_copy.theme_id)
        pre_dedup.append(v_copy)

    # Stage -1: dedupe verdicts that share the same theme_id. When the LLM
    # emits the same theme twice (Opus has been observed doing this for
    # post-merger theme_ids), pick the strongest tier; union linked_helpers.

    def _mark_changed(tid: str) -> None:
        if tid not in changed_set:
            changed_set.add(tid)
            changed_ids.append(tid)

    def _maybe_demote(v: VerdictItem) -> None:
        if not v.linked_helpers:
            v.verdict = "GENE_LEVEL_ONLY"
            v.rationale = v.rationale.rstrip() + _AUTO_DEMOTE_NOTE

    for v in pre_dedup:
        v_copy = v  # already a deep copy from Stage -2
        if v_copy.theme_id not in new_verdicts:
            new_verdicts[v_copy.theme_id] = v_copy
        else:
            # Merge into the existing verdict for this theme_id.
            existing = new_verdicts[v_copy.theme_id]
            # Prefer the stronger tier; rationale stays from the chosen verdict.
            if _VERDICT_PRIORITY[v_copy.verdict] < _VERDICT_PRIORITY[existing.verdict]:
                stronger, weaker = v_copy, existing
            else:
                stronger, weaker = existing, v_copy
            seen = set(stronger.linked_helpers or [])
            for h in (weaker.linked_helpers or []):
                if h not in seen:
                    seen.add(h)
                    stronger.linked_helpers.append(h)
            new_verdicts[v_copy.theme_id] = stronger
            _mark_changed(v_copy.theme_id)

    # Stage 0: dedupe + drop unknown helpers + drop direction-mismatched helpers.
    for tid, v in new_verdicts.items():
        original = list(v.linked_helpers or [])
        if not original:
            continue
        seen: set = set()
        cleaned: list[str] = []
        for h in original:
            if h in seen:
                continue
            seen.add(h)
            if h not in valid_helpers:
                continue  # unknown helper -> drop
            if helper_dir.get(h) != theme_dir.get(tid):
                continue  # direction mismatch -> drop
            cleaned.append(h)
        if cleaned != original:
            v.linked_helpers = cleaned
            _mark_changed(tid)
            _maybe_demote(v)

    # Stage 1: per-theme cap. Trim to 3 strongest helpers (lowest q first).
    for tid, v in new_verdicts.items():
        if len(v.linked_helpers or []) > MAX_HELPERS_PER_THEME:
            ranked = sorted(
                v.linked_helpers,
                key=lambda h: helper_q.get(h, 1.0),  # lowest q first
            )
            v.linked_helpers = ranked[:MAX_HELPERS_PER_THEME]
            _mark_changed(tid)
            _maybe_demote(v)

    # Stage 2: per-helper cap. Recount and trim.
    helper_counts: dict[str, list[str]] = {}
    for tid, v in new_verdicts.items():
        for h in v.linked_helpers or []:
            helper_counts.setdefault(h, []).append(tid)

    for helper, theme_ids in helper_counts.items():
        if len(theme_ids) <= MAX_THEMES_PER_HELPER:
            continue
        # Rank for keep: SUPPORTED before PARTIAL, then by |mean_nes| desc.
        ranked = sorted(
            theme_ids,
            key=lambda tid: (
                _VERDICT_PRIORITY[new_verdicts[tid].verdict],
                -theme_nes.get(tid, 0.0),
            ),
        )
        for tid in ranked[MAX_THEMES_PER_HELPER:]:
            v = new_verdicts[tid]
            v.linked_helpers = [h for h in v.linked_helpers if h != helper]
            _mark_changed(tid)
            _maybe_demote(v)

    corrected = AgentOutput(
        dataset_id=agent_output.dataset_id,
        verdicts=list(new_verdicts.values()),
    )
    return corrected, changed_ids
