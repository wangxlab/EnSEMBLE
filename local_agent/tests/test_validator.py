"""Unit tests for the v2.1 deterministic validator."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from local_agent.fallback import build_fallback_output  # noqa: E402
from local_agent.schemas import AgentInput, AgentOutput, Helper, Theme, VerdictItem  # noqa: E402
from local_agent.validator import (  # noqa: E402
    MAX_HELPERS_PER_THEME,
    MAX_THEMES_PER_HELPER,
    auto_correct_capacity,
    is_capacity_only_errors,
    validate_verdicts,
)


def _mk_input() -> AgentInput:
    helpers = [
        Helper(
            helper_name="ENCODE_MESENCHYMAL_STEM_CELL",
            helper_class="celltype",
            direction="UP",
            nes=0.46,
            q_value=0.04,
            leading_edge_n=28,
        ),
        Helper(
            helper_name="ENCODE_SKELETAL_MUSCLE_MYOBLAST",
            helper_class="celltype",
            direction="UP",
            nes=0.44,
            q_value=0.04,
            leading_edge_n=30,
        ),
        Helper(
            helper_name="ENCODE_FIBROBLAST",
            helper_class="celltype",
            direction="DOWN",
            nes=-0.40,
            q_value=0.03,
            leading_edge_n=20,
        ),
    ]
    themes = [
        Theme(
            theme_id="up_c1_emt",
            label="EMT",
            direction="UP",
            mean_nes=0.66,
            n_members=3,
            top_pathways=["HALLMARK_EMT"],
            top_leading_edge_genes=["VIM", "FN1"],
        ),
        Theme(
            theme_id="up_c2_e2f",
            label="E2F targets",
            direction="UP",
            mean_nes=0.88,
            n_members=3,
            top_pathways=["HALLMARK_E2F_TARGETS"],
            top_leading_edge_genes=["MCM2"],
        ),
        Theme(
            theme_id="dn_c1_ecm",
            label="ECM remodeling",
            direction="DOWN",
            mean_nes=-0.5,
            n_members=2,
            top_pathways=["NABA_MATRISOME"],
            top_leading_edge_genes=["COL1A1"],
        ),
    ]
    return AgentInput(
        dataset_id="TEST",
        biological_context="Hand-crafted test input.",
        helpers=helpers,
        themes=themes,
    )


def _mk_output(items):
    return AgentOutput(dataset_id="TEST", verdicts=items)


_LONG_RATIONALE = (
    "Mesenchymal stem cell and myoblast enhancer programs are the regulatory "
    "context for the epithelial-mesenchymal transition theme, both upregulated, "
    "with concordant biology consistent with active mesenchymal regulation."
)


def test_valid_multi_helper_output_passes():
    inp = _mk_input()
    out = _mk_output(
        [
            VerdictItem(
                theme_id="up_c1_emt",
                theme_label="EMT",
                verdict="SUPPORTED",
                linked_helpers=["ENCODE_MESENCHYMAL_STEM_CELL", "ENCODE_SKELETAL_MUSCLE_MYOBLAST"],
                rationale=_LONG_RATIONALE,
            ),
            VerdictItem(
                theme_id="up_c2_e2f",
                theme_label="E2F targets",
                verdict="GENE_LEVEL_ONLY",
                linked_helpers=[],
                rationale=_LONG_RATIONALE,
            ),
            VerdictItem(
                theme_id="dn_c1_ecm",
                theme_label="ECM remodeling",
                verdict="SUPPORTED",
                linked_helpers=["ENCODE_FIBROBLAST"],
                rationale=_LONG_RATIONALE,
            ),
        ]
    )
    errors = validate_verdicts(inp, out)
    assert errors == [], errors


def test_glo_with_helpers_caught():
    inp = _mk_input()
    out = build_fallback_output(inp)
    out.verdicts[0].linked_helpers = ["ENCODE_MESENCHYMAL_STEM_CELL"]
    errors = validate_verdicts(inp, out)
    assert any("GENE_LEVEL_ONLY" in e and "linked_helpers" in e for e in errors), errors


def test_supported_with_empty_helpers_caught():
    inp = _mk_input()
    out = build_fallback_output(inp)
    out.verdicts[0].verdict = "SUPPORTED"
    out.verdicts[0].linked_helpers = []
    errors = validate_verdicts(inp, out)
    assert any("linked_helpers is empty" in e for e in errors), errors


def test_unknown_helper_in_list_caught():
    inp = _mk_input()
    out = build_fallback_output(inp)
    out.verdicts[0].verdict = "SUPPORTED"
    out.verdicts[0].linked_helpers = ["ENCODE_MESENCHYMAL_STEM_CELL", "UNKNOWN_HELPER"]
    errors = validate_verdicts(inp, out)
    assert any("UNKNOWN_HELPER" in e and "not in input" in e for e in errors), errors


def test_direction_mismatch_in_list_caught():
    inp = _mk_input()
    out = build_fallback_output(inp)
    # up_c1_emt is UP; mixing in a DOWN helper should trigger
    out.verdicts[0].verdict = "SUPPORTED"
    out.verdicts[0].linked_helpers = ["ENCODE_MESENCHYMAL_STEM_CELL", "ENCODE_FIBROBLAST"]
    errors = validate_verdicts(inp, out)
    assert any("direction" in e and "ENCODE_FIBROBLAST" in e for e in errors), errors


def test_capacity_cap_5_enforced():
    helpers = [
        Helper(
            helper_name="HOG",
            helper_class="tfbs",
            direction="UP",
            nes=0.5,
            q_value=0.01,
            leading_edge_n=10,
        )
    ]
    themes = [
        Theme(
            theme_id=f"t{i}",
            label=f"T{i}",
            direction="UP",
            mean_nes=0.5,
            n_members=1,
            top_pathways=["X"],
            top_leading_edge_genes=["G"],
        )
        for i in range(7)
    ]
    inp = AgentInput(
        dataset_id="X", biological_context="x", helpers=helpers, themes=themes
    )
    out = _mk_output(
        [
            VerdictItem(
                theme_id=f"t{i}",
                theme_label=f"T{i}",
                verdict="SUPPORTED",
                linked_helpers=["HOG"],
                rationale=_LONG_RATIONALE,
            )
            for i in range(7)
        ]
    )
    errors = validate_verdicts(inp, out)
    assert any(f"max {MAX_THEMES_PER_HELPER}" in e for e in errors), errors
    assert is_capacity_only_errors(errors)


def test_capacity_only_classifier():
    assert is_capacity_only_errors(["Helper 'X' appears in 6 themes (max 5)"])
    assert not is_capacity_only_errors([])
    # Short rationale is now correctable (auto-demote to GLO with synthetic msg)
    assert is_capacity_only_errors(["t1: rationale has 5 words (min 15)"])
    # But other validation errors are not correctable.
    assert not is_capacity_only_errors(["Missing verdicts for themes: ['x']"])


def test_auto_correct_removes_helper_from_excess_themes():
    """7 themes all linked to one helper; after correction, 5 keep the helper,
    2 lose it. If those 2 had no other helpers, they become GLO."""
    helpers = [
        Helper(
            helper_name="HOG",
            helper_class="tfbs",
            direction="UP",
            nes=0.5,
            q_value=0.01,
            leading_edge_n=10,
        ),
        Helper(
            helper_name="OTHER",
            helper_class="tfbs",
            direction="UP",
            nes=0.4,
            q_value=0.05,
            leading_edge_n=8,
        ),
    ]
    themes = [
        Theme(
            theme_id=f"t{i}",
            label=f"T{i}",
            direction="UP",
            mean_nes=0.9 - i * 0.1,
            n_members=1,
            top_pathways=["X"],
            top_leading_edge_genes=["G"],
        )
        for i in range(7)
    ]
    inp = AgentInput(
        dataset_id="X", biological_context="x", helpers=helpers, themes=themes
    )

    # First 5 themes have HOG only; last 2 have HOG + OTHER (so they survive
    # losing HOG by keeping OTHER).
    verdicts = []
    for i in range(7):
        helpers_for_theme = ["HOG"] if i < 5 else ["HOG", "OTHER"]
        verdicts.append(
            VerdictItem(
                theme_id=f"t{i}",
                theme_label=f"T{i}",
                verdict="SUPPORTED",
                linked_helpers=helpers_for_theme,
                rationale=_LONG_RATIONALE,
            )
        )
    out = _mk_output(verdicts)

    initial = validate_verdicts(inp, out)
    assert is_capacity_only_errors(initial), initial

    corrected, changed = auto_correct_capacity(inp, out)

    # Top 5 by |NES| desc among SUPPORTED keep HOG: t0 (0.9), t1 (0.8), t2 (0.7),
    # t3 (0.6), t4 (0.5). t5 (0.4) and t6 (0.3) lose HOG. t5/t6 had OTHER too,
    # so they keep OTHER and stay SUPPORTED.
    by_id = {v.theme_id: v for v in corrected.verdicts}
    for i in range(5):
        assert "HOG" in by_id[f"t{i}"].linked_helpers
    for i in range(5, 7):
        assert "HOG" not in by_id[f"t{i}"].linked_helpers
        assert by_id[f"t{i}"].linked_helpers == ["OTHER"]
        assert by_id[f"t{i}"].verdict == "SUPPORTED"

    assert validate_verdicts(inp, corrected) == []


def test_auto_correct_demotes_to_glo_when_only_helper_was_over_cap():
    """If a verdict's only helper is the over-cap one, removing it -> GLO."""
    helpers = [
        Helper(
            helper_name="HOG",
            helper_class="tfbs",
            direction="UP",
            nes=0.5,
            q_value=0.01,
            leading_edge_n=10,
        )
    ]
    themes = [
        Theme(
            theme_id=f"t{i}",
            label=f"T{i}",
            direction="UP",
            mean_nes=0.9 - i * 0.1,
            n_members=1,
            top_pathways=["X"],
            top_leading_edge_genes=["G"],
        )
        for i in range(7)
    ]
    inp = AgentInput(
        dataset_id="X", biological_context="x", helpers=helpers, themes=themes
    )
    out = _mk_output(
        [
            VerdictItem(
                theme_id=f"t{i}",
                theme_label=f"T{i}",
                verdict="SUPPORTED",
                linked_helpers=["HOG"],
                rationale=_LONG_RATIONALE,
            )
            for i in range(7)
        ]
    )
    corrected, changed = auto_correct_capacity(inp, out)
    by_id = {v.theme_id: v for v in corrected.verdicts}
    for i in range(5):
        assert by_id[f"t{i}"].verdict == "SUPPORTED"
    for i in range(5, 7):
        assert by_id[f"t{i}"].verdict == "GENE_LEVEL_ONLY"
        assert by_id[f"t{i}"].linked_helpers == []
        assert "auto-demoted" in by_id[f"t{i}"].rationale
    assert validate_verdicts(inp, corrected) == []


def test_short_rationale_caught():
    inp = _mk_input()
    out = build_fallback_output(inp)
    out.verdicts[0].rationale = "too short"
    errors = validate_verdicts(inp, out)
    assert any("rationale has" in e for e in errors), errors


def test_fallback_is_valid():
    inp = _mk_input()
    out = build_fallback_output(inp)
    errors = validate_verdicts(inp, out)
    assert errors == [], errors


def test_missing_theme_caught():
    inp = _mk_input()
    out = _mk_output(
        [
            VerdictItem(
                theme_id="up_c1_emt",
                theme_label="EMT",
                verdict="GENE_LEVEL_ONLY",
                linked_helpers=[],
                rationale=_LONG_RATIONALE,
            )
        ]
    )
    errors = validate_verdicts(inp, out)
    assert any("Missing verdicts" in e for e in errors), errors


def test_auto_correct_dedupes_duplicate_theme_ids():
    """Opus has been observed emitting the same theme_id twice. The
    auto-correction should merge them, preferring the stronger tier and
    unioning linked_helpers."""
    inp = _mk_input()
    out = _mk_output(
        [
            VerdictItem(
                theme_id="up_c1_emt",
                theme_label="EMT",
                verdict="PARTIAL",
                linked_helpers=["ENCODE_MESENCHYMAL_STEM_CELL"],
                rationale=_LONG_RATIONALE,
            ),
            VerdictItem(
                theme_id="up_c1_emt",  # duplicate
                theme_label="EMT",
                verdict="SUPPORTED",  # stronger tier
                linked_helpers=["ENCODE_SKELETAL_MUSCLE_MYOBLAST"],
                rationale=_LONG_RATIONALE,
            ),
            VerdictItem(
                theme_id="up_c2_e2f",
                theme_label="E2F targets",
                verdict="GENE_LEVEL_ONLY",
                linked_helpers=[],
                rationale=_LONG_RATIONALE,
            ),
            VerdictItem(
                theme_id="dn_c1_ecm",
                theme_label="ECM remodeling",
                verdict="SUPPORTED",
                linked_helpers=["ENCODE_FIBROBLAST"],
                rationale=_LONG_RATIONALE,
            ),
        ]
    )
    initial = validate_verdicts(inp, out)
    assert any("Duplicate verdict for theme_id" in e for e in initial), initial
    assert is_capacity_only_errors(initial)

    corrected, changed = auto_correct_capacity(inp, out)
    assert "up_c1_emt" in changed
    by_id = {v.theme_id: v for v in corrected.verdicts}
    assert len(corrected.verdicts) == 3
    # Stronger tier wins (SUPPORTED), helpers unioned
    assert by_id["up_c1_emt"].verdict == "SUPPORTED"
    assert set(by_id["up_c1_emt"].linked_helpers) == {
        "ENCODE_MESENCHYMAL_STEM_CELL",
        "ENCODE_SKELETAL_MUSCLE_MYOBLAST",
    }
    assert validate_verdicts(inp, corrected) == []


def test_per_theme_cap_caught():
    """A theme listing >3 helpers must trigger the per-theme capacity rule."""
    # 4 valid UP helpers, all direction-matched to one theme
    helpers = [
        Helper(
            helper_name=f"H{i}",
            helper_class="celltype",
            direction="UP",
            nes=0.5,
            q_value=0.01,
            leading_edge_n=10,
        )
        for i in range(4)
    ]
    themes = [
        Theme(
            theme_id="t0",
            label="T0",
            direction="UP",
            mean_nes=0.5,
            n_members=1,
            top_pathways=["X"],
            top_leading_edge_genes=["G"],
        )
    ]
    inp = AgentInput(dataset_id="X", biological_context="x", helpers=helpers, themes=themes)
    out = _mk_output(
        [
            VerdictItem(
                theme_id="t0",
                theme_label="T0",
                verdict="SUPPORTED",
                linked_helpers=["H0", "H1", "H2", "H3"],
                rationale=_LONG_RATIONALE,
            )
        ]
    )
    errors = validate_verdicts(inp, out)
    assert any(f"max {MAX_HELPERS_PER_THEME}" in e for e in errors), errors
    assert is_capacity_only_errors(errors), errors


def test_per_theme_cap_auto_corrected_by_q_value():
    """Auto-correction trims to the 3 helpers with lowest q-value."""
    helpers = [
        Helper(helper_name="A", helper_class="celltype", direction="UP",
               nes=0.5, q_value=1e-50, leading_edge_n=10),  # strongest
        Helper(helper_name="B", helper_class="celltype", direction="UP",
               nes=0.5, q_value=1e-30, leading_edge_n=10),
        Helper(helper_name="C", helper_class="celltype", direction="UP",
               nes=0.5, q_value=1e-10, leading_edge_n=10),
        Helper(helper_name="D", helper_class="celltype", direction="UP",
               nes=0.5, q_value=1e-3,  leading_edge_n=10),  # weakest, will be dropped
    ]
    themes = [Theme(theme_id="t0", label="T0", direction="UP", mean_nes=0.5,
                    n_members=1, top_pathways=["X"], top_leading_edge_genes=["G"])]
    inp = AgentInput(dataset_id="X", biological_context="x", helpers=helpers, themes=themes)
    out = _mk_output([VerdictItem(
        theme_id="t0", theme_label="T0", verdict="SUPPORTED",
        linked_helpers=["A", "B", "C", "D"], rationale=_LONG_RATIONALE,
    )])
    corrected, changed = auto_correct_capacity(inp, out)
    by_id = {v.theme_id: v for v in corrected.verdicts}
    assert by_id["t0"].linked_helpers == ["A", "B", "C"], by_id["t0"].linked_helpers
    assert "t0" in changed
    assert validate_verdicts(inp, corrected) == []


def test_duplicate_helper_in_list_caught():
    inp = _mk_input()
    out = build_fallback_output(inp)
    out.verdicts[0].verdict = "SUPPORTED"
    out.verdicts[0].linked_helpers = ["ENCODE_MESENCHYMAL_STEM_CELL", "ENCODE_MESENCHYMAL_STEM_CELL"]
    errors = validate_verdicts(inp, out)
    assert any("duplicate helpers" in e for e in errors), errors


if __name__ == "__main__":
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(0 if failed == 0 else 1)
