"""Training-specific data types for SynthStats."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, NotRequired, Required, TypedDict

import torch


class SubTBBatch(TypedDict, total=False):
    """Typed contract for SubTB/TB training batches.

    Token-level or action-level batch structure for trajectory balance
    and SubTB losses.
    """

    log_probs: Required[torch.Tensor]  # [B, T]
    log_reward: Required[torch.Tensor]  # [B]
    loss_mask: Required[torch.Tensor]  # [B, T] bool
    entropy: Required[torch.Tensor]  # [B, T]
    ref_log_probs: NotRequired[torch.Tensor]  # [B, T]
    eos_logprobs: NotRequired[torch.Tensor]  # [B, T]
    granularity: NotRequired[Literal["token", "action"]]
    is_agentic: NotRequired[bool]


@dataclass
class AgentLoopOutput:
    """Output of a multi-turn episode: generated tokens, masks, reward and messages."""

    prompt_ids: list[int] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    reward: float = 0.0
    messages: list[dict[str, str]] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class BufferEntry:
    """Stores action sequence only (no tensors); re-scored on sample.

    log_reward retains the reward temperature used at collection time; changing
    the temperature does not rescale stored rewards.
    """

    actions: list[dict[str, Any]]
    log_reward: float
    observations: list[str]
    atif_id: str | None = None
    reward_source: str | None = None
    evidence_kind: str | None = None
    truth_owner: str | None = None
    reward_components: list[dict[str, Any]] = field(default_factory=list)
    scalarization: dict[str, Any] | None = None
    rubric: dict[str, Any] | None = None
    policy_version: int = 0
    temperature: float = 1.0
    artifacts: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": self.actions,
            "log_reward": self.log_reward,
            "observations": self.observations,
            "atif_id": self.atif_id,
            "reward_source": self.reward_source,
            "evidence_kind": self.evidence_kind,
            "truth_owner": self.truth_owner,
            "reward_components": list(self.reward_components),
            "scalarization": self.scalarization,
            "rubric": self.rubric,
            "policy_version": self.policy_version,
            "temperature": self.temperature,
            "artifacts": dict(self.artifacts),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BufferEntry:
        return cls(
            actions=data["actions"],
            log_reward=data["log_reward"],
            observations=data["observations"],
            atif_id=data.get("atif_id"),
            reward_source=data.get("reward_source"),
            evidence_kind=data.get("evidence_kind"),
            truth_owner=data.get("truth_owner"),
            reward_components=list(data.get("reward_components", [])),
            scalarization=data.get("scalarization"),
            rubric=data.get("rubric"),
            policy_version=data.get("policy_version", 0),
            temperature=data.get("temperature", 1.0),
            artifacts=dict(data.get("artifacts", {})),
            timestamp=data.get("timestamp", 0.0),
        )
