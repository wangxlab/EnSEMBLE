# Changelog

## 2.0.0 — Anthropic Claude agent rewrite

### Breaking changes

- The Python `local_agent/` module is rewritten end-to-end. The v1.x
  Gemini 4-stage pipeline (linker / evidence / critic / thesis) is
  replaced by a single-call Anthropic Claude evidence classifier
  (Sonnet/Opus 4.5) + optional mini-thesis writer.
- `--gsea-only` is removed. ESEA helpers are now required.
- Default temperature 0.3 → 0.0 (deterministic).
- Output artefact names changed; see [MIGRATION.md](MIGRATION.md) for
  the full mapping.
- `THEME_RULES_v1.0.yml` (YAML/regex taxonomy) and
  `resources/tf_helper_names.txt` deleted.

### Added

- `local_agent/cli.py` back-compat shim. Accepts v1.x flags
  (`--gsea-csv`, `--esea-csv`, `--background-txt`, …) and maps them to
  the v2.0 pipeline with a one-time deprecation notice per flag.
- `local_agent/runner.py` canonical entry: `--dataset NAME` style and
  `--gsea-csv / --esea-csv / --background-txt` file-path style.
- Output schema: `linked_helpers: List[str]` (multi-helper convergence),
  `theme_weight: float` (Stouffer-like `sum(-log10 q) / sqrt(n)`).
- Post-clustering merger (`local_agent.merger`): 0.5 leading-edge
  Jaccard threshold on full GSEA leading-edge sets, with
  HALLMARK-anchored representative selection to preserve specific
  themes (e.g. SNAI1 EMT cluster stays labelled EMT).
- Deterministic validator (`local_agent.validator`): 7 rules + 4-stage
  auto-correction (rationale length, theme-id dedup, direction mismatch
  surgical removal, per-theme/per-helper capacity caps).
- 3-run reproducibility check (`local_agent.reproducibility`) +
  consensus aggregator (`local_agent.consensus`) using verdict majority
  + helper union-then-intersection.
- Ground-truth scorer (`local_agent.evaluate`) with check primitives:
  `_check_supported_not_partial`, `_check_partial_not_supported`,
  `_check_weight_rank`, `_check_partial_gte_supported`.
- Report module (`local_agent.report`): three deterministic figures
  (compression bar, evidence network with GLO context, helper overview
  lollipop), markdown verdict table, mini-thesis caller + validator,
  weasyprint PDF assembler.
- 39 deterministic unit tests under `local_agent/tests/`.
- Package metadata: `__init__.py` exports, `__version__`,
  `pytest.ini`, `.gitignore`, `requirements.txt` refreshed.

### Changed

- `requirements.txt` no longer pins `google-generativeai`, `PyYAML`,
  `Pygments`, `seaborn`. Added `anthropic`, `pydantic`, `networkx`,
  `scipy`, `scikit-learn`, `dynamicTreeCut`.
- DESCRIPTION version 1.1.1 → 2.0.0.

### Migrated upstream files (from `ENSEMBLE_clustering_v1`)

- `local_agent/themes.py` (715-line locked clustering with two-bin
  auto-selection on N=250)
- `local_agent/prefilter.py`
- `local_agent/background.py`
- `local_agent/data_models.py` (minimal: GSEARecord, HelperRecord,
  Theme, ThemeSummary)

The R side (`R/*.R`, `inst/`, `man/`, `DESCRIPTION` imports) is
unchanged.

## 1.1.1 and earlier

See git history before tag `v2.0.0`. v1.x agent is preserved at the
repo root as `local_agent_v1_backup.tar.gz`.
