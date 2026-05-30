"""Mini-thesis lightweight validator.

Per mini_thesis_spec.md: thesis is prose, not structured. Validation logs
warnings (no fallback). Re-run is recommended only when a required section
is missing or hallucinated helpers are detected.
"""
from __future__ import annotations

import re
from typing import Iterable, List


REQUIRED_SECTIONS = [
    "### Enhancer-Supported Findings",
    "### Gene-Level Findings",
    "### Mechanistic Interpretation",
    "### Suggested Validation",
]

FORBIDDEN_PHRASES = [
    "interestingly",
    "notably",
    "it is worth noting",
    "ensemble is a framework",
    "our tool",
]

# Match helper-name-like tokens that look like they should be in input (avoid
# matching TF symbols like EP300 or POLR2A which never have these prefixes).
# Allow ., +, - inside; trailing punctuation is stripped post-match.
HELPER_PATTERN = re.compile(
    r"(?:ENCODE|eRNAbase|CATlas|dbSUPER|Ensembl)_[A-Za-z0-9._+-]+"
)
_TRAILING_PUNCT = ".,;:!?)"


def _normalize_helper_match(m: str) -> str:
    return m.rstrip(_TRAILING_PUNCT)


def validate_mini_thesis(text: str, helper_names: Iterable[str]) -> List[str]:
    """Return list of warning strings. Empty list = clean."""
    warnings: List[str] = []
    helper_set = set(helper_names)

    # 1. Length
    word_count = len(text.split())
    if word_count < 300:
        warnings.append(f"Too short: {word_count} words (min 300 per spec)")
    if word_count > 800:
        warnings.append(f"Too long: {word_count} words (max 800 per spec)")

    # 2. Required sections
    for section in REQUIRED_SECTIONS:
        if section not in text:
            warnings.append(f"Missing section: {section}")

    # 3. Bold summary at start
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if not lines or not lines[0].lstrip().startswith("**"):
        warnings.append("Missing bold summary sentence at start")

    # 4. Hallucinated helpers (strip trailing punctuation before matching)
    mentioned = {_normalize_helper_match(m) for m in HELPER_PATTERN.findall(text)}
    unknown = sorted(mentioned - helper_set)
    if unknown:
        warnings.append(f"Helper names in text not in input: {unknown}")

    # 5. Forbidden phrases
    lower = text.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in lower:
            warnings.append(f"Forbidden phrase found: '{phrase}'")

    return warnings
