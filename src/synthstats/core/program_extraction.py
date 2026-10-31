"""Utilities for extracting submitted programs from model text outputs."""

from __future__ import annotations

import json
import re

CODE_FENCE_PATTERN = re.compile(r"```(?:python|haskell)\s*(.*?)```", re.DOTALL)
SUBMIT_PROGRAM_PATTERN = re.compile(
    r'<submit_program(?:\s+language="[^"]*")?>(.*?)</submit_program>', re.DOTALL
)


def extract_from_tool_call(text: str) -> str | None:
    """Parse a JSON tool-call object and return the program code.

    Accepts the OpenAI/ATIF standard format:
        {"name": "submit_program", "arguments": {"code": "..."}}

    Returns None if the text is not a valid submit_program tool call.
    """
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(obj, dict):
        return None
    if obj.get("name") not in ("submit_program", "submit"):
        return None

    args = obj.get("arguments", {})
    if not isinstance(args, dict):
        return None

    code = args.get("code") or args.get("program") or args.get("content")
    return str(code).strip() if code else None


def extract_program_text(text: str) -> str | None:
    """Extract code payload.

    Priority order:
    1. JSON tool-call  {"name": "submit_program", "arguments": {"code": "..."}}
    2. XML tag         <submit_program>...</submit_program>
    3. Code fence      ```python ... ```
    """
    tool_call_result = extract_from_tool_call(text)
    if tool_call_result is not None:
        return tool_call_result

    submit_match = SUBMIT_PROGRAM_PATTERN.search(text)
    if submit_match:
        return submit_match.group(1).strip()

    code_match = CODE_FENCE_PATTERN.search(text)
    if code_match:
        return code_match.group(1).strip()

    return None


__all__ = [
    "CODE_FENCE_PATTERN",
    "SUBMIT_PROGRAM_PATTERN",
    "extract_from_tool_call",
    "extract_program_text",
]
