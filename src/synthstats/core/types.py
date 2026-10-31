"""Core types for SynthStats."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    reasoning_content: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.reasoning_content is not None:
            d["reasoning_content"] = self.reasoning_content
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            role=data["role"],
            content=data["content"],
            tool_call_id=data.get("tool_call_id"),
            reasoning_content=data.get("reasoning_content"),
        )


@dataclass
class Action:
    pass


@dataclass
class FinalAnswer(Action):
    text: str


@dataclass
class ToolCall(Action):
    name: str
    input: dict[str, Any]
    raw: str  # original text representation
    tool_call_id: str | None = None


@dataclass
class Program(Action):
    """Action to submit a probabilistic program."""

    code: str
    language: str = "pymc"  # "pymc" | "numpyro" | "lazyppl"


@dataclass
class StepResult:
    next_state: Any
    done: bool
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class Reward:
    """Multi-component reward signal."""

    total: float
    components: dict[str, float]
    info: dict[str, Any]
    factors: dict[str, float] = field(default_factory=dict)
    scalarization: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "components": self.components,
            "info": self.info,
            "factors": self.factors,
            "scalarization": self.scalarization,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Reward:
        return cls(
            total=data["total"],
            components=data["components"],
            info=data["info"],
            factors=data.get("factors", {}),
            scalarization=data.get("scalarization"),
        )


@dataclass
class Trajectory:
    """Complete episode data for training.

    messages covers ALL conversation turns (system, user, assistant, tool).
    token_ids/token_logprobs/loss_mask/eos_logprobs cover ONLY assistant generations
    (one entry per generate() call). Use len(token_ids) for generation count,
    len(messages) for full conversation length.
    """

    messages: list[Message]
    token_ids: list[list[int]]  # one entry per assistant generation
    token_logprobs: list[list[float]]  # one entry per assistant generation
    loss_mask: list[list[bool]]  # one entry per assistant generation
    reward: Reward
    # EOS log probabilities for SubTB flow matching (empty = vanilla TB)
    eos_logprobs: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [m.to_dict() for m in self.messages],
            "token_ids": self.token_ids,
            "token_logprobs": self.token_logprobs,
            "loss_mask": self.loss_mask,
            "reward": self.reward.to_dict(),
            "eos_logprobs": self.eos_logprobs,
        }

    @classmethod
    def empty_stub(cls, messages: list[Message]) -> Trajectory:
        """Stub with no token data, for judge scoring without generation info."""
        return cls(
            messages=messages,
            token_ids=[],
            token_logprobs=[],
            loss_mask=[],
            reward=Reward(0.0, {}, {}),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        return cls(
            messages=[Message.from_dict(m) for m in data["messages"]],
            token_ids=data["token_ids"],
            token_logprobs=data["token_logprobs"],
            loss_mask=data["loss_mask"],
            reward=Reward.from_dict(data["reward"]),
            eos_logprobs=data.get("eos_logprobs", []),
        )


@dataclass
class Transition:
    """Single LLM call in SkyRL-Agent format.

    Non-terminal transitions have reward=0.0 and done=False;
    the final transition carries the episode reward.
    """

    input_ids: list[int]
    output_ids: list[int]
    logprobs: list[float]
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)
