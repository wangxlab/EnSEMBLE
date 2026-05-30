"""Configuration constants for the EnSEMBLE agent."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# v2.0 layout: themes.py / prefilter.py / background.py are siblings of this
# file inside the same package; no sys.path hacks needed.
PACKAGE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = PACKAGE_DIR / "prompts"

# Default I/O roots — resolved relative to the user's current working
# directory at run time. The CLI / shim can override both.
INPUTS_DIR = Path("inputs")
OUTPUTS_DIR = Path("outputs")


@dataclass(frozen=True)
class AgentConfig:
    """Runtime parameters for the Anthropic Claude classifier + thesis."""

    model: str = "claude-sonnet-4-5"
    temperature: float = 0.0
    # 4096 truncated PANC1/iPSC responses (40 themes ~= 5-6k output tokens).
    # Sonnet/Opus 4.5 supports up to 64k. 8192 leaves comfortable headroom.
    max_tokens: int = 8192
    max_retries: int = 1
    gsea_q_threshold: float = 0.05
    # Theme caps applied during clustering. The upstream defaults (10 per
    # direction, 20 total) drop themes ground-truth depends on (e.g., BT20
    # EMT). 40 total / no per-direction cap keeps all biologically critical
    # themes in scope while staying within the token budget.
    theme_cap_per_direction: int = 0  # 0 = unlimited
    theme_cap_total: int = 40
    # v2.2: post-clustering merger threshold (leading-edge Jaccard on the
    # union of all member-pathway leading edges). 0.0 disables; 0.5 catches
    # biologically near-redundant themes without false positives.
    merge_jaccard_threshold: float = 0.5

    @property
    def api_key(self) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it before calling the agent."
            )
        return key


@dataclass(frozen=True)
class AnalysisSettings:
    """Thresholds and caps controlling deterministic filtering of GSEA
    pathways and the locked clustering pipeline.

    Locked parameters (do not change without re-validating clustering):
      - cluster_linkage = "average"
      - cluster_auto_params = True with cutpoint at N=250 pathways
      - Bin A (N < 250):  deepSplit=3, minClusterSize=5
      - Bin B (N >= 250): deepSplit=1, minClusterSize=10
    """

    gsea_top_n: int = 0
    gsea_q_threshold: float = 0.05
    gsea_min_per_direction: int = 1
    helper_q_threshold: float = 0.05
    helper_nes_threshold: float = 0.25
    helper_max_per_direction: int = 50
    helper_claims_per_theme: int = 0
    theme_cap: int = 10
    theme_top_pathways: int = 3
    theme_leading_edge_target: int = 12
    theme_cap_total: int = 20
    head_linker_theme_cap: int = 15
    head_linker_theme_cap_total: int = 30

    # --- locked clustering parameters (two-bin auto-selection) ---
    cluster_linkage: str = "average"
    cluster_deep_split: int = 1
    cluster_min_size: int = 10
    cluster_min_similarity: float = 0.0
    cluster_pam_stage: bool = True
    cluster_include_orphans: bool = True
    cluster_auto_params: bool = True
    cluster_bin_cutpoint_n: int = 250
    cluster_small_deep_split: int = 3
    cluster_small_min_size: int = 5
    cluster_large_deep_split: int = 1
    cluster_large_min_size: int = 10


DEFAULT_BACKGROUND_KEYS = (
    "Study_ID",
    "System_Model",
    "Perturbation",
    "Contrast",
    "Assay_Context",
    "Known_Biology",
    "Key_Questions",
    "Cell_Types_of_Interest_(optional)",
    "Expected_Phenotypes_or_Trends_(optional, describe expectations not mandates)",
    "Pathway_Hypotheses_(optional)",
    "Red_Flag_Contradictions",
)
