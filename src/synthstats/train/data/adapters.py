"""Data-format adapters for training integrations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from synthstats.core.types import Trajectory, Transition

if TYPE_CHECKING:
    from synthstats.train.data.collectors import CollectedTrajectory
    from synthstats.train.data.types import AgentLoopOutput

logger = logging.getLogger(__name__)


def trajectory_to_transitions(traj: Trajectory) -> list[Transition]:
    """One Transition per assistant generation. The last one carries the episode reward."""
    n_gens = len(traj.token_ids)
    if n_gens == 0:
        return []
    if len(traj.token_logprobs) != n_gens:
        raise ValueError(
            f"token_ids has {n_gens} entries but token_logprobs has {len(traj.token_logprobs)}"
        )

    transitions: list[Transition] = []

    for gen_idx in range(n_gens):
        output_ids = traj.token_ids[gen_idx]
        logprobs = traj.token_logprobs[gen_idx]

        is_last = gen_idx == n_gens - 1
        transitions.append(
            Transition(
                input_ids=[],
                output_ids=output_ids,
                logprobs=logprobs,
                reward=traj.reward.total if is_last else 0.0,
                done=is_last,
                info={"gen_idx": gen_idx},
            )
        )

    return transitions


def collected_trajectory_to_transitions(
    traj: CollectedTrajectory,
) -> list[Transition]:
    """Per-action log_probs mapped to single-element Transition logprobs."""
    from synthstats.train.data.collectors import CollectedTrajectory as _CT  # noqa: PLC0415

    if not isinstance(traj, _CT):
        raise TypeError(f"expected CollectedTrajectory, got {type(traj).__name__}")
    n_actions = len(traj.actions)
    if n_actions == 0:
        return []

    transitions: list[Transition] = []
    for i in range(n_actions):
        if i >= len(traj.log_probs):
            raise ValueError(
                f"log_probs has {len(traj.log_probs)} entries but trajectory "
                f"has {n_actions} actions (missing logprob at index {i})"
            )
        lp = traj.log_probs[i].item()
        is_last = i == n_actions - 1
        info = {"action_idx": i, "action": traj.actions[i]}
        if traj.artifacts:
            info["trajectory_artifacts"] = traj.artifacts
        transitions.append(
            Transition(
                input_ids=[],
                output_ids=[],
                logprobs=[lp],
                reward=traj.reward if is_last else 0.0,
                done=is_last,
                info=info,
            )
        )

    return transitions


def agent_loop_output_to_dict(output: AgentLoopOutput) -> dict[str, torch.Tensor]:
    """Convert AgentLoopOutput to a dict of tensors for batch construction."""
    return {
        "prompt_ids": torch.tensor(output.prompt_ids, dtype=torch.long),
        "response_ids": torch.tensor(output.response_ids, dtype=torch.long),
        "response_mask": torch.tensor(output.response_mask, dtype=torch.float),
        "reward": torch.tensor(output.reward, dtype=torch.float),
    }


def batch_agent_loop_outputs(
    outputs: list[AgentLoopOutput],
    *,
    pad_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Batch multiple AgentLoopOutputs into padded [B, T] tensors."""
    if not outputs:
        return {
            "response_ids": torch.zeros(0, 0, dtype=torch.long),
            "response_mask": torch.zeros(0, 0, dtype=torch.float),
            "rewards": torch.zeros(0, dtype=torch.float),
        }

    max_len = max(len(o.response_ids) for o in outputs)
    batch_size = len(outputs)

    response_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    response_mask = torch.zeros(batch_size, max_len, dtype=torch.float)
    rewards = torch.zeros(batch_size, dtype=torch.float)

    for i, output in enumerate(outputs):
        seq_len = len(output.response_ids)
        response_ids[i, :seq_len] = torch.tensor(output.response_ids, dtype=torch.long)
        response_mask[i, :seq_len] = torch.tensor(output.response_mask, dtype=torch.float)
        rewards[i] = output.reward

    return {
        "response_ids": response_ids,
        "response_mask": response_mask,
        "rewards": rewards,
    }


__all__ = [
    "agent_loop_output_to_dict",
    "batch_agent_loop_outputs",
    "collected_trajectory_to_transitions",
    "trajectory_to_transitions",
]
