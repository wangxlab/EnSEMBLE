"""Unit tests for post-clustering merger."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from local_agent.merger import jaccard, merge_redundant_themes  # noqa: E402
from local_agent.schemas import Theme  # noqa: E402


def _t(theme_id: str, direction: str, nes: float, genes: list[str]) -> Theme:
    return Theme(
        theme_id=theme_id,
        label=theme_id.replace("_", " "),
        direction=direction,
        mean_nes=nes,
        n_members=1,
        top_pathways=["X"],
        top_leading_edge_genes=genes,
    )


def test_jaccard_identical():
    assert jaccard(["A", "B", "C"], ["A", "B", "C"]) == 1.0


def test_jaccard_disjoint():
    assert jaccard(["A", "B"], ["C", "D"]) == 0.0


def test_jaccard_partial():
    # 2 shared / 4 union = 0.5
    assert abs(jaccard(["A", "B", "C"], ["B", "C", "D"]) - 0.5) < 1e-9


def test_no_merge_when_threshold_zero():
    themes = [
        _t("up_a", "UP", 0.8, ["G1", "G2"]),
        _t("up_b", "UP", 0.7, ["G1", "G2"]),
    ]
    merged, log = merge_redundant_themes(themes, jaccard_threshold=0.0)
    assert len(merged) == 2
    assert log.n_after_total == 2
    assert log.groups == []


def test_simple_merge_above_threshold():
    """Two themes share 2/3 genes -> Jaccard = 0.5, exactly threshold."""
    themes = [
        _t("up_a", "UP", 0.9, ["G1", "G2", "G3"]),
        _t("up_b", "UP", 0.7, ["G2", "G3", "G4"]),  # share G2, G3 -> J=0.5
    ]
    merged, log = merge_redundant_themes(themes, jaccard_threshold=0.5)
    assert len(merged) == 1
    rep = merged[0]
    assert rep.theme_id.startswith("up_a")  # higher |NES| wins
    assert rep.theme_id.endswith("_merged2")
    # Union of genes preserved
    assert set(rep.top_leading_edge_genes) == {"G1", "G2", "G3", "G4"}
    assert log.n_before_total == 2
    assert log.n_after_total == 1
    assert len(log.groups) == 1
    assert log.groups[0].representative_id == "up_a"


def test_no_merge_below_threshold():
    themes = [
        _t("up_a", "UP", 0.9, ["G1", "G2", "G3"]),
        _t("up_b", "UP", 0.7, ["G3", "G4", "G5"]),  # share G3 only -> J=1/5=0.2
    ]
    merged, log = merge_redundant_themes(themes, jaccard_threshold=0.5)
    assert len(merged) == 2
    assert log.groups == []


def test_no_cross_direction_merge():
    """UP and DOWN must never merge even with identical genes."""
    themes = [
        _t("up_a", "UP", 0.9, ["G1", "G2"]),
        _t("dn_a", "DOWN", -0.9, ["G1", "G2"]),  # same genes, opposite direction
    ]
    merged, log = merge_redundant_themes(themes, jaccard_threshold=0.5)
    assert len(merged) == 2  # never merged
    assert log.groups == []


def test_transitive_merge():
    """A-B and B-C overlap; A-C may not, but all 3 should merge transitively."""
    themes = [
        _t("up_a", "UP", 0.9, ["G1", "G2", "G3", "G4"]),
        _t("up_b", "UP", 0.8, ["G3", "G4", "G5", "G6"]),  # A-B share G3,G4 -> 2/6=0.33 < 0.5
        _t("up_c", "UP", 0.7, ["G5", "G6", "G7", "G8"]),  # B-C share G5,G6 -> 2/6=0.33 < 0.5
    ]
    # No edges meet 0.5 -> no merge
    merged, log = merge_redundant_themes(themes, jaccard_threshold=0.5)
    assert len(merged) == 3

    # Now make A-B and B-C meet 0.5 each.
    themes2 = [
        _t("up_a", "UP", 0.9, ["G1", "G2", "G3", "G4"]),
        _t("up_b", "UP", 0.8, ["G2", "G3", "G4", "G5"]),  # A-B share G2,G3,G4 / 5 = 0.6
        _t("up_c", "UP", 0.7, ["G3", "G4", "G5", "G6"]),  # B-C share G3,G4,G5 / 5 = 0.6
    ]
    # A-C share G3,G4 / 6 = 0.33 (below threshold) but transitive should merge.
    merged2, log2 = merge_redundant_themes(themes2, jaccard_threshold=0.5)
    assert len(merged2) == 1
    rep = merged2[0]
    assert rep.theme_id == "up_a_merged3"
    assert log2.groups[0].representative_id == "up_a"
    # Members include up_b and up_c
    member_ids = {m["theme_id"] for m in log2.groups[0].members}
    assert member_ids == {"up_b", "up_c"}


def test_representative_picked_by_strongest_nes():
    """When multiple themes merge, the rep has the strongest |NES|."""
    themes = [
        _t("up_a", "UP", 0.5, ["G1", "G2", "G3"]),
        _t("up_b", "UP", 0.9, ["G1", "G2", "G3"]),  # higher |NES|
    ]
    merged, log = merge_redundant_themes(themes, jaccard_threshold=0.5)
    assert log.groups[0].representative_id == "up_b"


def test_hallmark_anchored_theme_wins_rep_selection():
    """A theme with HALLMARK_* in top_pathways[0] should be chosen as
    representative even if its |mean_nes| is weaker than the other theme."""
    a = _t("up_protein_mod", "UP", 0.71, ["G1", "G2", "G3"])  # stronger |NES|
    a.top_pathways = ["GOBP_REGULATION_OF_PROTEIN_MODIFICATION"]
    b = _t("up_emt", "UP", 0.63, ["G1", "G2", "G3"])  # weaker |NES| but HALLMARK
    b.top_pathways = ["HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION", "GOBP_LOCOMOTION"]
    merged, log = merge_redundant_themes([a, b], jaccard_threshold=0.5)
    assert len(merged) == 1
    assert log.groups[0].representative_id == "up_emt", "EMT must win as rep"
    assert merged[0].label == b.label


if __name__ == "__main__":
    pass  # see below


def test_n_members_summed():
    themes = [
        _t("up_a", "UP", 0.9, ["G1", "G2", "G3"]),
        _t("up_b", "UP", 0.8, ["G1", "G2", "G3"]),
    ]
    themes[0].n_members = 5
    themes[1].n_members = 7
    merged, log = merge_redundant_themes(themes, jaccard_threshold=0.5)
    assert merged[0].n_members == 12


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
