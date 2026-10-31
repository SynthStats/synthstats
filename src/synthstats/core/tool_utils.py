"""Task-agnostic helpers for OpenAI-style tool schemas."""

from __future__ import annotations

from typing import Any

from synthstats.runtime.codecs import ToolSpec


def tool_schemas_to_toolspecs(schemas: list[dict[str, Any]]) -> list[ToolSpec]:
    """Convert OpenAI-style function schemas to ToolSpec objects."""
    specs = []
    for schema in schemas:
        fn = schema.get("function", schema)
        specs.append(
            ToolSpec(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters=fn.get("parameters", {}),
            )
        )
    return specs
