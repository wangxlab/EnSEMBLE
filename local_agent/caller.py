"""Claude API call + JSON parsing + retry."""
from __future__ import annotations

import json
import re
from typing import Optional, Tuple

from .config import AgentConfig, PROMPTS_DIR
from .schemas import AgentInput, AgentOutput


SYSTEM_PROMPT = (PROMPTS_DIR / "system_prompt.txt").read_text()
RETRY_NUDGE = "Return valid JSON only matching the schema. No prose, no code fences."


class APICallError(RuntimeError):
    """Raised when the API call fails after retries."""


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort JSON extraction. Strips fenced code blocks."""
    text = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first JSON object.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def call_claude(
    agent_input: AgentInput,
    config: AgentConfig,
) -> Tuple[Optional[AgentOutput], dict]:
    """Make one API call (with up to config.max_retries retries) and parse the result.

    Returns (parsed_output_or_None, log_dict). The log captures the raw request
    and response across all attempts.
    """
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise APICallError(
            "anthropic SDK not installed; pip install anthropic in your env"
        ) from e

    client = Anthropic(api_key=config.api_key)

    user_payload = agent_input.model_dump(mode="json")
    user_message = json.dumps(user_payload)

    log: dict = {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "system_prompt_chars": len(SYSTEM_PROMPT),
        "input_payload": user_payload,
        "attempts": [],
    }

    messages = [{"role": "user", "content": user_message}]
    parsed: Optional[AgentOutput] = None

    for attempt in range(config.max_retries + 1):
        try:
            response = client.messages.create(
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except Exception as e:  # network/auth/etc.
            log["attempts"].append({"attempt": attempt, "error": repr(e)})
            continue

        raw_text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        attempt_log = {
            "attempt": attempt,
            "stop_reason": getattr(response, "stop_reason", None),
            "usage": getattr(response, "usage", None).__dict__ if getattr(response, "usage", None) else None,
            "raw_text": raw_text,
        }
        log["attempts"].append(attempt_log)

        candidate = _extract_json(raw_text)
        if candidate is not None:
            try:
                parsed = AgentOutput.model_validate(candidate)
                attempt_log["parsed"] = True
                break
            except Exception as e:
                attempt_log["parsed"] = False
                attempt_log["parse_error"] = repr(e)
        else:
            attempt_log["parsed"] = False
            attempt_log["parse_error"] = "no JSON object found in response"

        # Retry: append assistant's malformed response and a nudge.
        messages = messages + [
            {"role": "assistant", "content": raw_text},
            {"role": "user", "content": RETRY_NUDGE},
        ]

    return parsed, log
