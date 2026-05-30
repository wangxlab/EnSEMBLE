"""Unit tests for the mini-thesis validator."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from local_agent.report.thesis_validator import validate_mini_thesis  # noqa: E402


_VALID_TEMPLATE = """**Summary sentence in bold here for testing.**

### Enhancer-Supported Findings
The Epithelial-mesenchymal transition theme (UP, mean NES +0.66) is
SUPPORTED by ENCODE_MESENCHYMAL_STEM_CELL helper. Leading edges include
ZEB1 and VIM and FN1 and CDH1.

### Gene-Level Findings
Cell cycle and proliferation themes are GENE_LEVEL_ONLY because the
mesenchymal helpers do not regulate replication machinery.

### Mechanistic Interpretation
Enhancer evidence singles out EMT as the primary regulatory event, while
the dominant transcriptomic response is downstream proliferation.

### Suggested Validation
Western blot for E-cadherin and Vimentin would test the EMT verdict.
"""


def _pad(text: str, target_words: int = 320) -> str:
    """Pad to the minimum word count by repeating filler words."""
    have = len(text.split())
    if have < target_words:
        text = text + "\n\n" + ("filler word " * (target_words - have))
    return text


def test_valid_text_returns_no_warnings():
    text = _pad(_VALID_TEMPLATE)
    helpers = ["ENCODE_MESENCHYMAL_STEM_CELL"]
    warns = validate_mini_thesis(text, helpers)
    # Allow length warnings if our pad missed; section/hallucination should pass
    section_warns = [w for w in warns if "Missing section" in w or "bold summary" in w or "not in input" in w or "Forbidden phrase" in w]
    assert section_warns == [], section_warns


def test_too_short_caught():
    text = "**Summary**\n\n### Enhancer-Supported Findings\nx"
    warns = validate_mini_thesis(text, [])
    assert any("Too short" in w for w in warns)


def test_missing_section_caught():
    text = _pad(_VALID_TEMPLATE.replace("### Suggested Validation\n", "### Other\n"))
    warns = validate_mini_thesis(text, ["ENCODE_MESENCHYMAL_STEM_CELL"])
    assert any("Missing section" in w and "Suggested Validation" in w for w in warns)


def test_missing_bold_summary_caught():
    text = _pad(_VALID_TEMPLATE.replace("**Summary sentence in bold here for testing.**", "Plain text summary"))
    warns = validate_mini_thesis(text, ["ENCODE_MESENCHYMAL_STEM_CELL"])
    assert any("bold summary" in w for w in warns)


def test_hallucinated_helper_caught():
    text = _pad(_VALID_TEMPLATE + "\nThe ENCODE_FAKE_CELLTYPE helper appears here.")
    warns = validate_mini_thesis(text, ["ENCODE_MESENCHYMAL_STEM_CELL"])
    assert any("not in input" in w and "ENCODE_FAKE_CELLTYPE" in w for w in warns)


def test_forbidden_phrase_caught():
    text = _pad(_VALID_TEMPLATE + "\nInterestingly, this matters.")
    warns = validate_mini_thesis(text, ["ENCODE_MESENCHYMAL_STEM_CELL"])
    assert any("Forbidden phrase" in w and "interestingly" in w for w in warns)


def test_known_tfbs_helper_not_flagged():
    """TF symbols like SUPT5H or POLR2A don't match the prefix pattern,
    so they shouldn't be falsely flagged as hallucinations."""
    text = _pad(_VALID_TEMPLATE + "\nSUPT5H is a transcription elongation factor.")
    warns = validate_mini_thesis(text, ["ENCODE_MESENCHYMAL_STEM_CELL"])
    # The validator only checks prefixed helper names; TF symbols are out of scope.
    assert not any("SUPT5H" in w for w in warns)


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
