"""Unit tests for compute_theme_weight."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from local_agent.schemas import Helper, VerdictItem  # noqa: E402
from local_agent.weight import compute_theme_weight  # noqa: E402


def _h(name: str, q: float) -> Helper:
    return Helper(
        helper_name=name,
        helper_class="tfbs",
        direction="UP",
        nes=0.5,
        q_value=q,
        leading_edge_n=10,
    )


def test_glo_returns_zero():
    v = VerdictItem(
        theme_id="t",
        theme_label="T",
        verdict="GENE_LEVEL_ONLY",
        linked_helpers=[],
        rationale="x" * 100,
    )
    assert compute_theme_weight(v, {}) == 0.0


def test_supported_one_helper():
    v = VerdictItem(
        theme_id="t",
        theme_label="T",
        verdict="SUPPORTED",
        linked_helpers=["MSC"],
        rationale="x" * 100,
    )
    lookup = {"MSC": _h("MSC", 0.045)}
    expected = -math.log10(0.045) / math.sqrt(1)
    assert abs(compute_theme_weight(v, lookup) - expected) < 1e-9


def test_supported_two_helpers_stouffer():
    """BT20 EMT example: MSC (q=0.045) + MYOBLAST (q=0.045) -> ~1.91"""
    v = VerdictItem(
        theme_id="t",
        theme_label="T",
        verdict="SUPPORTED",
        linked_helpers=["MSC", "MYOBLAST"],
        rationale="x" * 100,
    )
    lookup = {"MSC": _h("MSC", 0.045), "MYOBLAST": _h("MYOBLAST", 0.045)}
    expected = (-math.log10(0.045) * 2) / math.sqrt(2)
    assert abs(compute_theme_weight(v, lookup) - expected) < 1e-9
    # Sanity vs spec example
    assert abs(compute_theme_weight(v, lookup) - 1.905) < 0.01


def test_partial_discount():
    v = VerdictItem(
        theme_id="t",
        theme_label="T",
        verdict="PARTIAL",
        linked_helpers=["MSC"],
        rationale="x" * 100,
    )
    lookup = {"MSC": _h("MSC", 0.045)}
    expected = (-math.log10(0.045) / math.sqrt(1)) * 0.5
    assert abs(compute_theme_weight(v, lookup) - expected) < 1e-9


def test_three_helper_convergence_per_spec():
    """iPSC synaptic example from the spec: q={1.6e-20, 4.7e-20, 9.8e-16} -> ~31.2"""
    v = VerdictItem(
        theme_id="t",
        theme_label="T",
        verdict="SUPPORTED",
        linked_helpers=["A", "B", "C"],
        rationale="x" * 100,
    )
    lookup = {
        "A": _h("A", 1.6e-20),
        "B": _h("B", 4.7e-20),
        "C": _h("C", 9.8e-16),
    }
    weight = compute_theme_weight(v, lookup)
    assert 30.5 < weight < 31.8, weight


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
