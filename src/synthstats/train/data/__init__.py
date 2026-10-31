"""Algorithm-specific data contracts for training."""

from synthstats.train.data.adapters import (
    agent_loop_output_to_dict,
    batch_agent_loop_outputs,
    collected_trajectory_to_transitions,
    trajectory_to_transitions,
)
from synthstats.train.data.types import AgentLoopOutput

__all__ = [
    "AgentLoopOutput",
    "agent_loop_output_to_dict",
    "batch_agent_loop_outputs",
    "collected_trajectory_to_transitions",
    "trajectory_to_transitions",
]
