"""Mini-thesis API caller (single Claude call per dataset).

Per mini_thesis_spec.md:
  - Sonnet 4.5 default (cost-efficient; thesis is prose, not classification)
  - Temperature 0.3 (slight variation acceptable for narrative)
  - Single call, no multi-turn
  - On missing-section: 1 retry with appended directive
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Tuple

from ..config import PROMPTS_DIR


THESIS_PROMPT = (PROMPTS_DIR / "thesis_prompt.txt").read_text()

DEFAULT_MODEL = "claude-sonnet-4-5"
# Spec said 2048 but datasets with many SUPPORTED+PARTIAL verdicts (iPSC has
# 11+7 = 18 paragraphs) exceed that. 4096 leaves comfortable headroom.
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.3

REQUIRED_SECTIONS = [
    "### Enhancer-Supported Findings",
    "### Gene-Level Findings",
    "### Mechanistic Interpretation",
    "### Suggested Validation",
]


class ThesisAPIError(RuntimeError):
    pass


def _strip_fences(text: str) -> str:
    text = text.strip()
    fenced = re.match(r"^```(?:\w+)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text


def _missing_sections(text: str) -> list:
    return [s for s in REQUIRED_SECTIONS if s not in text]


def generate_mini_thesis(
    thesis_input: dict,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = 1,
) -> Tuple[str, dict]:
    """Generate the mini-thesis markdown. Returns (text, log).

    The log captures model, params, and per-attempt usage / raw text.
    """
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise ThesisAPIError(
            "anthropic SDK not installed; pip install anthropic"
        ) from e

    import os
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ThesisAPIError("ANTHROPIC_API_KEY not set")
    client = Anthropic(api_key=key)

    user_payload = json.dumps(thesis_input)
    log: dict = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "input_payload_chars": len(user_payload),
        "attempts": [],
    }

    messages = [{"role": "user", "content": user_payload}]
    text: Optional[str] = None

    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=THESIS_PROMPT,
                messages=messages,
            )
        except Exception as e:
            log["attempts"].append({"attempt": attempt, "error": repr(e)})
            continue

        raw = "".join(
            b.text for b in response.content if getattr(b, "type", "") == "text"
        )
        cleaned = _strip_fences(raw)
        attempt_log = {
            "attempt": attempt,
            "stop_reason": getattr(response, "stop_reason", None),
            "usage": getattr(response, "usage", None).__dict__
                if getattr(response, "usage", None) else None,
            "raw_chars": len(raw),
            "cleaned_chars": len(cleaned),
        }
        log["attempts"].append(attempt_log)

        missing = _missing_sections(cleaned)
        if not missing:
            text = cleaned
            attempt_log["accepted"] = True
            break

        attempt_log["accepted"] = False
        attempt_log["missing_sections"] = missing
        # Retry with a corrective nudge
        if attempt < max_retries:
            nudge = (
                "Your previous response is missing the following required "
                f"section(s): {', '.join(missing)}. Please regenerate the "
                "complete report with all required sections in order, "
                "starting with the bold summary."
            )
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": nudge},
            ]
        else:
            # Use what we have despite missing sections.
            text = cleaned

    if text is None:
        raise ThesisAPIError("API call failed after all retries")

    return text, log


def build_thesis_input(agent_input: dict, verdicts: dict) -> dict:
    """Merge agent_input.json + verdicts.json into the thesis input shape."""
    return {
        "dataset_id": agent_input["dataset_id"],
        "biological_context": agent_input["biological_context"],
        "helpers": agent_input["helpers"],
        "themes": agent_input["themes"],
        "verdicts": verdicts["verdicts"],
    }
