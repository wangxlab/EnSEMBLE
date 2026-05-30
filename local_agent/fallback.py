"""Deterministic fallback: every theme classified as GENE_LEVEL_ONLY.

Used when the API call fails or output cannot be validated after one retry.
Per agent_rebuild_plan_v2.md Phase 2 spec.
"""
from __future__ import annotations

from .schemas import AgentInput, AgentOutput, VerdictItem


FALLBACK_RATIONALE = (
    "Fallback verdict: classification failed at the API or validation layer, "
    "so this theme is conservatively labeled GENE_LEVEL_ONLY pending re-run."
)


def build_fallback_output(agent_input: AgentInput) -> AgentOutput:
    verdicts = [
        VerdictItem(
            theme_id=t.theme_id,
            theme_label=t.label,
            verdict="GENE_LEVEL_ONLY",
            linked_helpers=[],
            rationale=FALLBACK_RATIONALE,
        )
        for t in agent_input.themes
    ]
    return AgentOutput(dataset_id=agent_input.dataset_id, verdicts=verdicts)
