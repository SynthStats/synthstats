"""Tool-schema definitions for agentic training integrations."""

from __future__ import annotations

from typing import Any

SYNTHSTATS_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query",
            "description": (
                "Run an experiment by querying the environment with specific "
                "parameter values. Returns observed data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Experiment specification or parameter values to query.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_eig",
            "description": (
                "Compute the Expected Information Gain for a proposed experiment. "
                "Helps decide which experiment to run next."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment": {
                        "type": "string",
                        "description": "Experiment specification to evaluate.",
                    },
                },
                "required": ["experiment"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_program",
            "description": (
                "Submit a probabilistic program (PyMC model) as the final answer. "
                "This terminates the episode and returns a reward based on the "
                "program's predictive quality (ELPD-LOO)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "PyMC program source code.",
                    },
                    "language": {
                        "type": "string",
                        "enum": ["pymc", "numpyro"],
                        "description": "Programming framework. Defaults to pymc.",
                    },
                },
                "required": ["code"],
            },
        },
    },
]

__all__ = ["SYNTHSTATS_TOOLS"]
