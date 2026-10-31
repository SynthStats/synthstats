"""OpenAI-style tool schemas for statistical modelling task environments."""

from __future__ import annotations

from typing import Any

from synthstats.core.tool_utils import tool_schemas_to_toolspecs as tool_schemas_to_toolspecs

STATISTICAL_MODELING_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query",
            "description": "Run an experiment to collect data from the environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Experiment query string, e.g. 'age=3'",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit a probabilistic program. Terminates the episode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The probabilistic program source code",
                    },
                    "language": {
                        "type": "string",
                        "enum": ["pymc", "lazyppl"],
                        "default": "pymc",
                        "description": "Programming language of the submitted program",
                    },
                },
                "required": ["code"],
            },
        },
    },
]


def get_tool_schemas(env_name: str = "dugongs") -> list[dict[str, Any]]:
    """Return tool schemas with per-environment query description."""
    import copy

    schemas = copy.deepcopy(STATISTICAL_MODELING_TOOLS)
    query_desc = {
        "dugongs": "Experiment query, e.g. 'age=3' to observe a dugong at that age",
        "peregrines": "Experiment query, e.g. 'year=2005' to observe falcon population",
        "eight_schools": "Experiment query, e.g. 'school=1' to observe treatment effect",
        "surgical": "Experiment query, e.g. 'hospital=1' to observe survival data",
    }
    desc = query_desc.get(env_name, "Experiment query string")
    schemas[0]["function"]["parameters"]["properties"]["query"]["description"] = desc
    return schemas


__all__ = [
    "STATISTICAL_MODELING_TOOLS",
    "get_tool_schemas",
    "tool_schemas_to_toolspecs",
]
