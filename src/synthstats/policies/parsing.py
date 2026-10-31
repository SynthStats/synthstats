"""Shared action parsing and utilities for policy implementations.

All policies parse LLM output through the same cascade:
inspect the full text for tool calls/submissions → parse a cleaned copy for
JSON fallback → return a runtime action dict.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_SUBMIT_PROGRAM_RE = re.compile(
    r"<submit_program(?:\s+language=['\"](?P<language>[^'\"]+)['\"])?[^>]*>"
    r"(?P<code>.*?)</submit_program>",
    re.DOTALL,
)
_JSON_DECODER = json.JSONDecoder()


def load_tokenizer(model_name: str) -> Any | None:
    """Load a HuggingFace tokenizer, returning None on failure.

    Handles two soft-failure cases:
    - transformers not installed (optional dep)
    - model not found locally or on Hub
    """
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name)
    except ImportError:
        logger.warning("transformers not installed; tokenizer unavailable")
        return None
    except (OSError, ValueError):
        logger.warning("could not load tokenizer for %s; using fallback", model_name)
        return None


def _strip_think_blocks(text: str) -> str:
    return _THINK_BLOCK_RE.sub("", text).strip()


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    for match in re.finditer(r"\{", text):
        try:
            candidate, _ = _JSON_DECODER.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    return None


def parse_action_dict(text: str) -> dict[str, Any]:
    raw_text = text.strip()

    # Hermes-style tool call: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    tc_match = re.search(r"<tool_call>(.*?)</tool_call>", raw_text, re.DOTALL)
    if tc_match:
        try:
            data = json.loads(tc_match.group(1).strip())
            # map submit tool to submission format
            if data.get("name") in ("submit", "submit_program"):
                args = data.get("arguments", {})
                return {
                    "type": "submit_program",
                    "payload": args.get("code", ""),
                    "language": args.get("language", "pymc"),
                }
            return data
        except json.JSONDecodeError:
            pass

    submit_match = _SUBMIT_PROGRAM_RE.search(raw_text)
    if submit_match:
        return {
            "type": "submit_program",
            "payload": submit_match.group("code").strip(),
            "language": submit_match.group("language") or "pymc",
        }

    cleaned_text = _strip_think_blocks(raw_text)
    candidate = _extract_first_json_object(cleaned_text)
    if candidate is not None:
        return candidate

    return {"type": "answer", "payload": cleaned_text or raw_text}


# alias used by hf_policy
parse_action = parse_action_dict


def render_action(action: dict[str, Any]) -> str:
    return json.dumps(action)


def mean_negative_logprob(logprobs: list[float]) -> float:
    if not logprobs:
        return 0.0
    return -sum(logprobs) / len(logprobs)


def build_prompt(obs: str) -> str:
    return (
        "You are an agent that responds to observations.\n"
        f"Observation: {obs}\n"
        "Respond with a JSON action: "
    )
