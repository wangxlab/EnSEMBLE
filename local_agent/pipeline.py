"""Thin backward-compatibility wrapper around the v2.0 runner.

The v1.x agent exposed ``run_pipeline`` as the primary entry point. v2.0
keeps the symbol so older callers (e.g. third-party scripts that imported
``from local_agent.pipeline import run_pipeline``) continue to work, with a
single semantic change: the agent now uses Anthropic Claude rather than
Google Gemini, and several v1.x arguments are accepted-but-ignored.

For new code, prefer ``local_agent.runner.run_dataset`` /
``local_agent.runner.run_from_files`` directly.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from .config import AgentConfig
from .runner import run_from_files


def run_pipeline(
    *,
    gsea_csv: Path,
    background_txt: Path,
    output_dir: Path,
    esea_csv: Optional[Path] = None,
    # v1.x kwargs accepted-but-ignored
    llm: Any = None,
    critic_llm: Any = None,
    analysis_settings: Any = None,
    gsea_only: bool = False,
    # v2.0 explicit overrides
    config: Optional[AgentConfig] = None,
    use_api: bool = True,
    **kwargs: Any,
) -> SimpleNamespace:
    """Run the end-to-end pipeline.

    Returns a SimpleNamespace with .helper_claims (== verdict dicts) and
    .helpers_available (== bool(esea_csv)) to match the v1.x return shape.

    Removed kwargs (raise on use):
        - gsea_only=True: no longer supported

    Ignored kwargs (warn on use):
        - llm, critic_llm, analysis_settings: v1.x Gemini constructs
    """
    if gsea_only:
        raise ValueError(
            "v2.0 removed gsea_only mode. ESEA helpers are required for "
            "classification."
        )
    if esea_csv is None:
        raise ValueError("v2.0 requires esea_csv; pass --esea-csv on the CLI.")

    for name, val in (("llm", llm), ("critic_llm", critic_llm), ("analysis_settings", analysis_settings)):
        if val is not None:
            warnings.warn(
                f"run_pipeline({name}=...) is ignored in v2.0; use AgentConfig instead.",
                DeprecationWarning,
                stacklevel=2,
            )
    for unknown in kwargs:
        warnings.warn(
            f"run_pipeline({unknown}=...) is unknown in v2.0 and was ignored.",
            DeprecationWarning,
            stacklevel=2,
        )

    cfg = config or AgentConfig()
    agent_output = run_from_files(
        gsea_csv=gsea_csv,
        esea_csv=esea_csv,
        background_txt=background_txt,
        config=cfg,
        output_dir=output_dir,
        use_api=use_api,
    )

    return SimpleNamespace(
        helper_claims=[v.model_dump(mode="json") for v in agent_output.verdicts],
        helpers_available=esea_csv is not None,
        dataset_id=agent_output.dataset_id,
        agent_output=agent_output,
    )


__all__ = ["run_pipeline"]
