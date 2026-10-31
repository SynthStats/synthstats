"""Policy protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from synthstats.core.constants import DEFAULT_MAX_NEW_TOKENS
from synthstats.core.types import Message

if TYPE_CHECKING:
    import torch

type PolicyOutput = tuple[dict[str, Any], float | torch.Tensor, float | torch.Tensor]


@dataclass
class GenConfig:
    max_tokens: int | None = DEFAULT_MAX_NEW_TOKENS
    temperature: float = 1.0
    top_p: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)


@dataclass
class Generation:
    text: str
    token_ids: list[int]
    token_logprobs: list[float]
    finish_reason: str  # "stop" | "length" | "tool_call"
    eos_logprobs: list[float] = field(default_factory=list)  # log p(EOS) per step, for SubTB


@dataclass
class TokenLogProbs:
    token_ids: list[int]
    logprobs: list[float]


@runtime_checkable
class Policy(Protocol):
    """Sampling interface: all policies implement this.

    Used by ``SkyRLRunner``, Harbor episode runners, and
    ``TrajectoryCollector`` for API-backed and local models alike.
    """

    def __call__(
        self,
        obs: str,
        temperature: float | None = None,
    ) -> PolicyOutput:
        """Sample an action. Returns (action_dict, log_prob, entropy)."""
        ...

    def sample_with_eos(
        self,
        obs: str,
        temperature: float | None = None,
    ) -> tuple[
        dict[str, Any],
        float | torch.Tensor,
        float | torch.Tensor,
        float | torch.Tensor | None,
    ]:
        """Returns (action_dict, log_prob, entropy, eos_logprob_or_None)."""
        ...

    def score_action(
        self,
        obs: str,
        action: dict[str, Any],
    ) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        """Returns (log_prob, entropy)."""
        ...

    def score_action_with_eos(
        self,
        obs: str,
        action: dict[str, Any],
        temperature: float | None = None,
    ) -> tuple[
        float | torch.Tensor,
        float | torch.Tensor,
        float | torch.Tensor | None,
    ]:
        """Returns (log_prob, entropy, eos_logprob_or_None)."""
        ...


@runtime_checkable
class GenerativePolicy(Policy, Protocol):
    """Full-sequence interface: extends Policy with token-level generation.

    Used by ``rollout_episode()`` for local HF models.  Requires the
    policy to produce full token-level generations and logprobs.
    Implemented by ``HFPolicy``.
    """

    def generate(self, messages: list[Message], *, gen: GenConfig) -> Generation: ...

    def logprobs(self, messages: list[Message], tokens: list[int]) -> TokenLogProbs: ...
