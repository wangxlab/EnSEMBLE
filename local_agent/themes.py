"""Theme grouping and ranking logic.

This module implements a configurable, regex-driven matcher that maps enriched
pathways to biological themes with prioritization and weighting. The rules are
defined in ``THEME_RULES_v1.0.yml`` and support canonical anchors, synonyms,
negative patterns, and cross-theme contributions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Dict, Iterable, List, Mapping, Tuple

try:  # optional dependency
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback when PyYAML missing
    yaml = None  # type: ignore

from .config import AnalysisSettings
from .data_models import GSEARecord, Theme, ThemeSummary


@dataclass(frozen=True)
class PatternDefinition:
    """Compiled regex pattern with matching metadata."""

    regex: re.Pattern[str]
    weight: float
    targets: Tuple[str, ...]


@dataclass
class ThemeRule:
    """Declarative rule describing how to match a theme."""

    identifier: str
    name: str
    priority: int
    canonical_patterns: Tuple[PatternDefinition, ...]
    positive_patterns: Tuple[PatternDefinition, ...]
    negative_patterns: Tuple[PatternDefinition, ...]
    contributions: Mapping[str, float]
    seed_sets: Tuple[str, ...] = ()
    notes: str | None = None

    def evaluate(self, context: Mapping[str, str]) -> float:
        """Return the aggregate weight contributed by this rule."""

        if self.negative_patterns and any(_pattern_matches(pat, context) for pat in self.negative_patterns):
            return 0.0
        canonical_score = sum(
            pattern.weight for pattern in self.canonical_patterns if _pattern_matches(pattern, context)
        )
        positive_score = sum(
            pattern.weight for pattern in self.positive_patterns if _pattern_matches(pattern, context)
        )
        total = canonical_score + positive_score
        if canonical_score and not positive_score:
            # Encourage canonical anchors even when synonyms are missing.
            total += canonical_score * 0.25
        return total


@dataclass
class ThemeAccumulator:
    """Collects weighted matches for a theme."""

    label: str
    priority: int
    members: Dict[GSEARecord, float] = field(default_factory=dict)

    def add(self, record: GSEARecord, weight: float) -> None:
        if weight <= 0:
            return
        self.members[record] = self.members.get(record, 0.0) + weight

    @property
    def total_weight(self) -> float:
        return sum(self.members.values())

    @property
    def total_signal(self) -> float:
        return sum(record.score * weight for record, weight in self.members.items())

    @property
    def ranking_score(self) -> float:
        if not self.members:
            return 0.0
        base = self.total_signal / max(self.total_weight, 1e-6)
        return base * (1.0 + self.priority / 100.0)

    def to_theme(self, direction: str) -> Theme:
        ordered_records = sorted(self.members.keys(), key=lambda rec: rec.score, reverse=True)
        return Theme(label=self.label, direction=direction, terms=ordered_records)


@dataclass(frozen=True)
class ThemeConfig:
    """Container for parsed theme configuration."""

    metadata: Mapping[str, object]
    defaults: Mapping[str, object]
    rules: Tuple[ThemeRule, ...]
    priorities: Mapping[str, int]


def normalize_term(term: str) -> str:
    """Return a normalized representation for backward compatibility."""

    return _prepare_text_variants(term).get("underscored", "")


def match_theme(term: str) -> Dict[str, float]:
    """Match a raw term string against configured themes.

    Returns a mapping of theme label to aggregate weight. This helper is
    maintained for legacy callers; new code should use :func:`match_record`.
    """

    dummy_record = GSEARecord(
        term=term,
        source=None,
        nes=0.0,
        q_value=1.0,
        size=0,
        direction="",
        score=0.0,
        leading_edge=(),
    )
    return match_record(dummy_record)


def match_record(record: GSEARecord) -> Dict[str, float]:
    """Compute weighted theme contributions for a GSEA record."""

    context = _build_context(record)
    contributions: Dict[str, float] = {}
    for rule in THEME_RULES:
        weight = rule.evaluate(context)
        if weight <= 0:
            continue
        for theme_name, factor in rule.contributions.items():
            contributions[theme_name] = contributions.get(theme_name, 0.0) + weight * factor
    return contributions


def group_by_theme(records: List[GSEARecord], direction: str, settings: AnalysisSettings) -> List[Theme]:
    accumulators: Dict[str, ThemeAccumulator] = {}
    fallback: List[GSEARecord] = []

    for record in records:
        matches = match_record(record)
        if not matches:
            fallback.append(record)
            continue
        for label, weight in matches.items():
            priority = THEME_PRIORITIES.get(label, THEME_PRIORITIES.get("__default__", 50))
            accumulator = accumulators.setdefault(label, ThemeAccumulator(label=label, priority=priority))
            accumulator.add(record, weight)

    ranked_accumulators = sorted(accumulators.values(), key=lambda acc: acc.ranking_score, reverse=True)
    selected = ranked_accumulators[: settings.theme_cap]
    themes = [acc.to_theme(direction) for acc in selected if acc.members]

    fallback.sort(key=lambda rec: rec.score, reverse=True)
    suffix = 1
    while fallback and len(themes) < settings.theme_cap:
        record = fallback.pop(0)
        themes.append(Theme(label=f"Additional {direction} signal {suffix}: {record.term}", direction=direction, terms=[record]))
        suffix += 1

    if fallback:
        if themes:
            themes[-1].terms.extend(fallback)
        else:
            themes.append(Theme(label=f"Additional {direction} signals", direction=direction, terms=fallback))

    return themes[: settings.theme_cap]


def _slugify(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-")
    return slug.lower() or "theme"


def _collect_leading_edges(records: List[GSEARecord], max_genes: int) -> tuple[str, ...]:
    counter: Counter[str] = Counter()
    order: Dict[str, int] = {}
    ticker = 0
    for rec in records:
        for gene in rec.leading_edge:
            canonical = gene.strip()
            if not canonical:
                continue
            counter[canonical] += 1
            if canonical not in order:
                order[canonical] = ticker
                ticker += 1
    if not counter:
        return tuple()
    ranked = sorted(counter.items(), key=lambda item: (-item[1], order.get(item[0], 0)))
    genes = [gene for gene, _ in ranked[:max_genes]]
    if len(genes) < max_genes:
        fallback = sorted(order.items(), key=lambda kv: kv[1])
        for gene, _ in fallback:
            if gene not in genes:
                genes.append(gene)
            if len(genes) >= max_genes:
                break
    return tuple(genes[:max_genes])


def build_theme_summaries(
    records_by_direction: Dict[str, List[GSEARecord]],
    settings: AnalysisSettings,
    output_dir: "Path | None" = None,
) -> Dict[str, List[ThemeSummary]]:
    """Group GSEA records into themes via leading-edge Jaccard clustering.

    This replaces the v6 regex/YAML taxonomy. The return shape is unchanged
    (``Dict[direction, List[ThemeSummary]]``) so every downstream LLM stage
    continues to work without modification.

    Parameters
    ----------
    records_by_direction
        Mapping "UP"/"DOWN" -> list of records that already passed the
        pipeline's qcut (``AnalysisSettings.gsea_q_threshold``) in
        ``prefilter.split_gsea_by_direction``.
    settings
        Clustering knobs live on ``AnalysisSettings`` (``cluster_deep_split``,
        ``cluster_min_size``, ``cluster_min_similarity``,
        ``cluster_include_orphans``, ``cluster_linkage``, ``cluster_pam_stage``).
    output_dir
        When provided, the clustering diagnostics (per-direction
        ``cluster_themes_<dir>.json`` + ``cluster_themes_<dir>_network.png/.pdf``
        + combined ``cluster_themes.json``) are written here. The returned
        ``ThemeSummary`` objects are unchanged.
    """
    # Local imports to keep the legacy section of this file import-light.
    from .cluster_themes import cluster_records  # noqa: WPS433
    import json as _json  # noqa: WPS433
    from pathlib import Path as _Path  # noqa: WPS433

    summaries: Dict[str, List[ThemeSummary]] = {"UP": [], "DOWN": []}
    max_genes = min(15, max(10, settings.theme_leading_edge_target))
    combined_audit: Dict[str, dict] = {}

    for direction, records in records_by_direction.items():
        if not records:
            summaries[direction] = []
            continue

        # Auto-select (deepSplit, minClusterSize) from N using the two-bin
        # lookup locked by the parameter grid sweep. Grid rationale:
        # outputs/grid_sweep/ shows no single combo produces K in [8,18] for
        # all 8 direction-units; this 2-bin rule hits 8/8.
        if settings.cluster_auto_params:
            if len(records) < settings.cluster_bin_cutpoint_n:
                ds = settings.cluster_small_deep_split
                mcs = settings.cluster_small_min_size
            else:
                ds = settings.cluster_large_deep_split
                mcs = settings.cluster_large_min_size
        else:
            ds = settings.cluster_deep_split
            mcs = settings.cluster_min_size

        result = cluster_records(
            records,
            linkage_method=settings.cluster_linkage,
            deep_split=ds,
            min_cluster_size=mcs,
            min_similarity=settings.cluster_min_similarity,
            pam_stage=settings.cluster_pam_stage,
        )

        # Index records by term for fast lookup back to GSEARecord objects.
        term_to_record = {rec.term: rec for rec in records}

        dir_summaries: List[ThemeSummary] = []
        for theme in result.themes:
            # Sort member GSEARecords by intra-cluster score: higher |NES| first,
            # then lower q-value. Keep top pathways for downstream LLM context.
            member_recs = [term_to_record[m.term] for m in theme.members
                           if m.term in term_to_record]
            if not member_recs:
                continue
            top_terms = sorted(
                member_recs, key=lambda r: r.score, reverse=True
            )[: settings.theme_top_pathways]
            effect = sum(r.nes for r in top_terms) / len(top_terms)
            q_value = min(r.q_value for r in top_terms)
            collections = [r.source for r in top_terms if r.source]
            collection = collections[0] if collections else None
            leading_edges = _collect_leading_edges(member_recs, max_genes)
            theme_id = f"{direction.lower()}_c{theme.cluster}_{_slugify(theme.label)}"
            dir_summaries.append(
                ThemeSummary(
                    theme_id=theme_id,
                    label=theme.label,
                    direction=direction,
                    collection=collection,
                    effect=effect,
                    q_value=q_value,
                    top_pathways=top_terms,
                    leading_edges=leading_edges,
                )
            )

        # Optionally pass orphans through as singleton ThemeSummaries so the
        # LLM still sees strong-but-unique signals. (Mirrors v6's fallback
        # "Additional signal N: TERM_NAME" — but explicitly marked as orphan.)
        if settings.cluster_include_orphans and result.orphans:
            for orphan in result.orphans:
                rec = term_to_record.get(orphan.term)
                if rec is None:
                    continue
                clean = _clean_term_label(orphan.term)
                leading_edges = _collect_leading_edges([rec], max_genes)
                theme_id = (
                    f"{direction.lower()}_orphan_{_slugify(clean)}"
                )
                dir_summaries.append(
                    ThemeSummary(
                        theme_id=theme_id,
                        label=f"Orphan: {clean}",
                        direction=direction,
                        collection=rec.source,
                        effect=rec.nes,
                        q_value=rec.q_value,
                        top_pathways=[rec],
                        leading_edges=leading_edges,
                    )
                )

        # Rank themes by |mean NES| desc, tiebreak by min q-value asc.
        # Cluster id order (by size) is retained in the audit JSON, but the
        # LLM budget is spent on the themes with the largest effect size.
        dir_summaries.sort(key=lambda t: (-abs(t.effect), t.q_value))
        dir_cap = settings.theme_cap if settings.theme_cap > 0 else len(dir_summaries)
        summaries[direction] = dir_summaries[:dir_cap] if dir_cap else dir_summaries

        # Emit audit artifacts per direction.
        if output_dir is not None:
            out = _Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            audit = result.to_json_dict()
            audit["direction"] = direction
            audit["n_input_records"] = len(records)
            combined_audit[direction] = audit
            with (out / f"cluster_themes_{direction.lower()}.json").open("w") as fh:
                _json.dump(audit, fh, indent=2)
            if result.records:
                try:
                    from .plot_theme_network import plot_clustered_network  # noqa: WPS433
                    import matplotlib.pyplot as _plt  # noqa: WPS433
                    bin_tag = (
                        "auto:binA" if settings.cluster_auto_params
                        and len(records) < settings.cluster_bin_cutpoint_n
                        else "auto:binB" if settings.cluster_auto_params
                        else "manual"
                    )
                    plot_clustered_network(
                        result,
                        out_png=str(out / f"cluster_themes_{direction.lower()}_network.png"),
                        out_pdf=str(out / f"cluster_themes_{direction.lower()}_network.pdf"),
                        title=(
                            f"{direction} themes - leading-edge Jaccard + "
                            f"dynamicTreeCut (n={len(result.records)}, "
                            f"deepSplit={ds}, minSize={mcs}, {bin_tag})"
                        ),
                    )
                    _plt.close("all")
                except Exception as exc:  # pragma: no cover - plotting is best-effort
                    print(f"[WARN] cluster network plot failed for {direction}: {exc}")

    # Honor the global theme_cap_total by trimming least-effective themes.
    total_cap = max(1, settings.theme_cap_total)
    while (len(summaries["UP"]) + len(summaries["DOWN"])) > total_cap:
        direction = "UP" if len(summaries["UP"]) >= len(summaries["DOWN"]) and summaries["UP"] else "DOWN"
        if not summaries[direction]:
            break
        remove_idx = min(
            range(len(summaries[direction])),
            key=lambda i: abs(summaries[direction][i].effect),
        )
        summaries[direction].pop(remove_idx)

    # Combined audit file for convenience.
    if output_dir is not None and combined_audit:
        out = _Path(output_dir)
        payload = {
            "params": next(iter(combined_audit.values())).get("params", {}),
            "directions": combined_audit,
            "final_themes_by_direction": {
                d: [
                    {
                        "theme_id": t.theme_id,
                        "label": t.label,
                        "direction": t.direction,
                        "effect": t.effect,
                        "q_value": t.q_value,
                        "n_top_pathways": len(t.top_pathways),
                        "leading_edges": list(t.leading_edges),
                    }
                    for t in summaries[d]
                ]
                for d in ("UP", "DOWN")
            },
        }
        with (out / "cluster_themes.json").open("w") as fh:
            _json.dump(payload, fh, indent=2)

    return summaries


def _clean_term_label(term: str, maxlen: int = 48) -> str:
    cleaned = re.sub(
        r"^(REACTOME_|GOBP_|GOCC_|GOMF_|HALLMARK_|KEGG_|WP_|PID_|BIOCARTA_)",
        "",
        term,
    )
    cleaned = cleaned.replace("_", " ").strip()
    tokens = cleaned.split()
    pretty: List[str] = []
    for tok in tokens:
        if tok.isupper() and len(tok) <= 4:
            pretty.append(tok)
        else:
            pretty.append(tok.title())
    label = " ".join(pretty)
    if len(label) > maxlen:
        label = label[: maxlen - 1].rstrip() + "\u2026"
    return label


# Internal helpers -----------------------------------------------------------------


def _legacy_theme_config() -> ThemeConfig:
    legacy_map: Dict[str, Iterable[str]] = {
        "TNF–NF-κB / inflammatory": ["TNF", "NFkB", "INFLAM", "TNFA"],
        "IFN / JAK-STAT": ["INTERFERON", "IFN", "JAK", "STAT"],
        "E2F / MYC cell cycle": ["E2F", "MYC", "S_PHASE", "CELL_CYCLE"],
        "DNA replication & repair": ["DNA_REP", "DNA REPLICATION", "REPAIR", "CHK"],
        "G2/M checkpoint": ["G2M", "MITOTIC", "CHROMOSOME", "SEGREG"],
        "Apoptosis / p53": ["P53", "APOP", "DNA_DAMAGE_RESPONSE"],
        "TNF stress & apoptosis": ["TNFA", "DEATH", "CASP"],
        "EMT / ECM / adhesion": ["EMT", "EXTRACELL", "ECM", "ADHESION", "MATRIX", "MESENCHYM", "EPITHEL"],
        "Lineage / differentiation": ["ERYTH", "MYELOID", "LYMPH", "STEM", "DIFFERENT"],
        "Metabolism": ["METAB", "OXIDATIVE", "MITO", "GLYCOL"],
    }

    rules: List[ThemeRule] = []
    priorities: Dict[str, int] = {"__default__": 50}
    for idx, (name, keywords) in enumerate(legacy_map.items(), start=1):
        patterns = tuple(
            PatternDefinition(
                regex=re.compile(keyword, re.IGNORECASE),
                weight=3.0,
                targets=("underscored", "spaced", "raw"),
            )
            for keyword in keywords
        )
        rule = ThemeRule(
            identifier=f"legacy_{idx}",
            name=name,
            priority=50,
            canonical_patterns=patterns,
            positive_patterns=(),
            negative_patterns=(),
            contributions={name: 1.0},
        )
        rules.append(rule)
        priorities[name] = 50

    metadata = {"source": "legacy", "note": "PyYAML not available; using built-in theme rules."}
    defaults: Dict[str, object] = {}
    return ThemeConfig(metadata=metadata, defaults=defaults, rules=tuple(rules), priorities=priorities)


def _load_theme_config(path: Path) -> ThemeConfig:
    if yaml is None:  # pragma: no cover - fallback when PyYAML missing
        return _legacy_theme_config()
    if not path.exists():
        return _legacy_theme_config()
    with path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    if not isinstance(raw_config, dict):
        raise ValueError("Theme configuration must deserialize to a mapping.")

    metadata = {
        key: raw_config.get(key)
        for key in ("version", "updated", "changelog")
        if key in raw_config
    }
    defaults = raw_config.get("defaults", {}) or {}
    default_priority = int(defaults.get("priority", 50))
    default_targets = tuple(defaults.get("targets", ("spaced", "underscored", "raw")))
    weight_defaults = defaults.get("weights", {}) or {}
    canonical_default_weight = float(weight_defaults.get("canonical", 3.0))
    positive_default_weight = float(weight_defaults.get("positive", 1.0))
    negative_targets = tuple(defaults.get("negative_targets", default_targets))

    raw_themes = raw_config.get("themes")
    if not isinstance(raw_themes, list):
        raise ValueError("Theme configuration must provide a 'themes' list.")

    rules: List[ThemeRule] = []
    priorities: Dict[str, int] = {"__default__": default_priority}

    for index, theme_entry in enumerate(raw_themes):
        if not isinstance(theme_entry, dict):
            raise ValueError("Each theme entry must be a mapping.")

        identifier = theme_entry.get("id") or f"theme_{index}"
        name = theme_entry.get("name")
        if not name:
            raise ValueError(f"Theme entry {identifier} is missing a 'name'.")
        priority = int(theme_entry.get("priority", default_priority))
        priorities[name] = priority

        contribution_cfg = theme_entry.get("contributions", {}) or {}
        primary_weight = float(contribution_cfg.get("primary", 1.0))
        contributions: Dict[str, float] = {name: primary_weight}
        for secondary in contribution_cfg.get("secondary", []) or []:
            if not isinstance(secondary, dict) or "theme" not in secondary:
                raise ValueError(f"Theme {identifier} has malformed secondary contribution: {secondary}")
            target = secondary["theme"]
            weight = float(secondary.get("weight", 0.0))
            contributions[target] = contributions.get(target, 0.0) + weight
            priorities.setdefault(target, default_priority)

        patterns_cfg = theme_entry.get("patterns", {}) or {}
        canonical_patterns = _compile_patterns(
            patterns_cfg.get("canonical", []),
            default_weight=canonical_default_weight,
            default_targets=default_targets,
        )
        positive_patterns = _compile_patterns(
            patterns_cfg.get("positive", []),
            default_weight=positive_default_weight,
            default_targets=default_targets,
        )
        negative_patterns = _compile_patterns(
            patterns_cfg.get("negative", []),
            default_weight=0.0,
            default_targets=negative_targets,
        )

        seed_sets = tuple(patterns_cfg.get("seed_sets", []) or theme_entry.get("seed_sets", []) or [])
        notes = theme_entry.get("notes")

        rules.append(
            ThemeRule(
                identifier=identifier,
                name=name,
                priority=priority,
                canonical_patterns=canonical_patterns,
                positive_patterns=positive_patterns,
                negative_patterns=negative_patterns,
                contributions=contributions,
                seed_sets=tuple(seed_sets),
                notes=notes,
            )
        )

    return ThemeConfig(metadata=metadata, defaults=defaults, rules=tuple(rules), priorities=priorities)


def _compile_patterns(patterns: Iterable[object], default_weight: float, default_targets: Tuple[str, ...]) -> Tuple[PatternDefinition, ...]:
    compiled: List[PatternDefinition] = []
    for entry in patterns:
        if isinstance(entry, str):
            pattern_str = entry
            weight = default_weight
            targets = default_targets
        elif isinstance(entry, dict):
            pattern_str = entry.get("pattern")
            if not pattern_str:
                raise ValueError("Pattern entries must include a 'pattern' key or be bare strings.")
            weight = float(entry.get("weight", default_weight))
            entry_targets = entry.get("targets")
            if isinstance(entry_targets, str):
                targets = (entry_targets,)
            elif isinstance(entry_targets, Iterable):
                targets = tuple(entry_targets)
            else:
                targets = default_targets
        else:
            raise ValueError(f"Unsupported pattern entry: {entry}")
        # Collapse doubled backslashes introduced by YAML single-quoted strings so
        # regex escape sequences such as ``\b`` work as intended.
        pattern_str = pattern_str.replace("\\\\", "\\")
        compiled.append(
            PatternDefinition(regex=re.compile(pattern_str, re.IGNORECASE), weight=weight, targets=targets)
        )
    return tuple(compiled)


def _pattern_matches(pattern: PatternDefinition, context: Mapping[str, str]) -> bool:
    for target in pattern.targets:
        text = context.get(target)
        if text and pattern.regex.search(text):
            return True
    return False


def _build_context(record: GSEARecord) -> Dict[str, str]:
    term_variants = _prepare_text_variants(record.term)
    context: Dict[str, str] = {
        "raw": term_variants["raw"],
        "spaced": term_variants["spaced"],
        "underscored": term_variants["underscored"],
        "compact": term_variants["compact"],
    }

    source_text = record.source or ""
    source_variants = _prepare_text_variants(source_text) if source_text else {"spaced": "", "underscored": "", "raw": "", "compact": ""}
    context["source"] = source_variants.get("spaced", "")
    context["source_raw"] = source_variants.get("raw", "")
    context["combined"] = " ".join(filter(None, (context["source"], context["spaced"]))).strip()

    description = getattr(record, "description", "") or ""
    if description:
        description_variants = _prepare_text_variants(description)
        context["description"] = description_variants["spaced"]
    else:
        context["description"] = ""

    return context


def _prepare_text_variants(value: str | None) -> Dict[str, str]:
    text = (value or "").strip()
    folded = _fold_to_ascii(text.upper())
    raw = folded
    spaced = _DASH_PATTERN.sub(" ", folded)
    spaced = _SPACE_PATTERN.sub(" ", spaced).strip()
    underscored = spaced.replace(" ", "_")
    compact = underscored.replace("_", "")
    return {"raw": raw, "spaced": spaced, "underscored": underscored, "compact": compact}


_GREEK_MAP = {
    "Α": "A",
    "Β": "B",
    "Γ": "G",
    "Δ": "D",
    "Ε": "E",
    "Ζ": "Z",
    "Η": "H",
    "Θ": "TH",
    "Ι": "I",
    "Κ": "K",
    "Λ": "L",
    "Μ": "M",
    "Ν": "N",
    "Ξ": "X",
    "Ο": "O",
    "Π": "P",
    "Ρ": "R",
    "Σ": "S",
    "Τ": "T",
    "Υ": "Y",
    "Φ": "PH",
    "Χ": "CH",
    "Ψ": "PS",
    "Ω": "O",
    "α": "A",
    "β": "B",
    "γ": "G",
    "δ": "D",
    "ε": "E",
    "ζ": "Z",
    "η": "H",
    "θ": "TH",
    "ι": "I",
    "κ": "K",
    "λ": "L",
    "μ": "M",
    "ν": "N",
    "ξ": "X",
    "ο": "O",
    "π": "P",
    "ρ": "R",
    "σ": "S",
    "ς": "S",
    "τ": "T",
    "υ": "Y",
    "φ": "PH",
    "χ": "CH",
    "ψ": "PS",
    "ω": "O",
}

_DASH_PATTERN = re.compile(r"[-–—/_]+")
_SPACE_PATTERN = re.compile(r"\s+")


def _fold_to_ascii(value: str) -> str:
    if not value:
        return ""
    return "".join(_GREEK_MAP.get(ch, ch) for ch in value)


THEME_RULES_PATH = Path(__file__).with_name("THEME_RULES_v1.0.yml")
THEME_CONFIG = _load_theme_config(THEME_RULES_PATH)
THEME_RULES = THEME_CONFIG.rules
THEME_PRIORITIES = THEME_CONFIG.priorities
