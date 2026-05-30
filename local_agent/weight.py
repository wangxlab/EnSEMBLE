"""Theme evidence weight (v2.1).

Stouffer-like aggregation of helper q-values:
    weight = sum(-log10(helper_q)) / sqrt(n_linked_helpers)

Discount of 0.5x is applied for PARTIAL verdicts (vs SUPPORTED).
GENE_LEVEL_ONLY themes return 0.0 (no helpers, no evidence weight).

The weight is computed post-classification and stored on each VerdictItem
for downstream use (network node sizing, theme ranking, report tables).
"""
from __future__ import annotations

import math
from typing import Iterable

from .schemas import AgentInput, AgentOutput, VerdictItem


PARTIAL_DISCOUNT = 0.5
_Q_FLOOR = 1e-300


def compute_theme_weight(verdict: VerdictItem, helpers_lookup: dict) -> float:
    """Compute Stouffer-like evidence weight for one verdict."""
    if verdict.verdict == "GENE_LEVEL_ONLY" or not verdict.linked_helpers:
        return 0.0

    log_q_sum = 0.0
    n = 0
    for h in verdict.linked_helpers:
        meta = helpers_lookup.get(h)
        if meta is None:
            continue
        q = meta.q_value if hasattr(meta, "q_value") else meta.get("q_value", 1.0)
        log_q_sum += -math.log10(max(q, _Q_FLOOR))
        n += 1

    if n == 0:
        return 0.0

    raw_weight = log_q_sum / math.sqrt(n)
    if verdict.verdict == "PARTIAL":
        raw_weight *= PARTIAL_DISCOUNT

    return raw_weight


def annotate_weights(agent_input: AgentInput, agent_output: AgentOutput) -> AgentOutput:
    """Set verdict.theme_weight on every verdict in agent_output, in place."""
    helpers_lookup = {h.helper_name: h for h in agent_input.helpers}
    for v in agent_output.verdicts:
        v.theme_weight = round(compute_theme_weight(v, helpers_lookup), 4)
    return agent_output
