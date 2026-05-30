"""Pydantic models for EnSEMBLE agent input and output JSON.

Schemas are defined in agent_rebuild_plan_v2.md Part 1.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


Direction = Literal["UP", "DOWN"]
Verdict = Literal["SUPPORTED", "PARTIAL", "GENE_LEVEL_ONLY"]


class Helper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    helper_name: str
    helper_class: str
    direction: Direction
    nes: float
    q_value: float
    leading_edge_n: int = Field(ge=0)
    top_hallmark: Optional[str] = None


class Theme(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme_id: str
    label: str
    direction: Direction
    mean_nes: float
    n_members: int = Field(ge=1)
    top_pathways: List[str]
    top_leading_edge_genes: List[str]


class AgentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    biological_context: str
    helpers: List[Helper]
    themes: List[Theme]


class VerdictItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme_id: str
    theme_label: str
    verdict: Verdict
    # v2.1: multi-helper support. Empty list for GENE_LEVEL_ONLY; one or more
    # helper names for SUPPORTED/PARTIAL.
    linked_helpers: List[str] = Field(default_factory=list)
    rationale: str
    # Optional Stouffer-like evidence weight, computed post-classification.
    theme_weight: Optional[float] = None
    # Optional metadata: which consensus rule produced this verdict (1-3 in v2.1).
    consensus_rule: Optional[int] = None


class AgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    verdicts: List[VerdictItem]
