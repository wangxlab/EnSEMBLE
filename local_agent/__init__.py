"""EnSEMBLE evidence-classifier agent (v2.0, Anthropic Claude).

Public API:

    from local_agent import run_pipeline               # v1.x-compatible entry
    from local_agent.runner import run_dataset, run_from_files
    from local_agent.config import AgentConfig, AnalysisSettings
    from local_agent.schemas import AgentInput, AgentOutput, VerdictItem

CLI entry points:

    python -m local_agent.cli                 # back-compat shim (old + new flags)
    python -m local_agent.runner              # canonical runner
    python -m local_agent.consensus           # 3-run consensus
    python -m local_agent.evaluate            # ground-truth scoring
    python -m local_agent.reproducibility     # 3x reproducibility check
    python -m local_agent.report.build_thesis # mini-thesis generation
    python -m local_agent.report.build_report # final PDF report
"""
from .pipeline import run_pipeline
from .runner import run_dataset, run_from_files
from .config import AgentConfig, AnalysisSettings

__version__ = "2.0.0"
__all__ = [
    "run_pipeline",
    "run_dataset",
    "run_from_files",
    "AgentConfig",
    "AnalysisSettings",
    "__version__",
]
