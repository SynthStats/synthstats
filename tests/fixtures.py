"""Shared test fixtures."""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from synthstats.core.policy import GenConfig, Generation, PolicyOutput, TokenLogProbs
from synthstats.core.types import Message, Reward, StepResult, Trajectory
from synthstats.policies.hf_policy import _EOSPolicyMixin


@dataclass
class ToyState:
    step: int = 0
    done: bool = False


class ToyTask:
    name = "toy"

    def reset(self, seed: int | None = None) -> ToyState:
        if seed is not None:
            random.seed(seed)
        return ToyState(step=0, done=False)

    def observe(self, state: ToyState) -> list[Message]:
        return [
            Message(role="system", content="You are a test agent."),
            Message(role="user", content="Please provide a JSON answer."),
        ]

    def step(self, state: ToyState, action: Any) -> StepResult:
        return StepResult(
            next_state=ToyState(step=state.step + 1, done=True),
            done=True,
            artifacts={"completed": True},
        )


class ToyJudge:
    def score(self, *, task_name: str, trajectory: Trajectory, artifacts: dict[str, Any]) -> Reward:
        return Reward(
            total=1.0,
            components={"base": 1.0},
            info={"task": task_name},
        )


class MockHFPolicy(_EOSPolicyMixin):
    """Mock policy for testing without loading actual models."""

    def __init__(
        self,
        require_grad_logp: bool = False,
        device: str = "cpu",
        fixed_text: str = "This is a mock response.",
        fixed_token_ids: list[int] | None = None,
        fixed_token_logprobs: list[float] | None = None,
        fixed_eos_logprobs: list[float] | None = None,
        fixed_finish_reason: str = "stop",
    ) -> None:
        self.require_grad_logp = require_grad_logp
        self.device = device

        self._fixed_text = fixed_text
        self._fixed_token_ids = fixed_token_ids or [101, 102, 103, 104, 105]
        self._fixed_token_logprobs = fixed_token_logprobs or [-0.1, -0.2, -0.15, -0.25, -0.1]
        self._fixed_finish_reason = fixed_finish_reason

        if fixed_eos_logprobs is not None:
            self._fixed_eos_logprobs = fixed_eos_logprobs
        else:
            # starts low (unlikely to stop), increases toward end
            n = len(self._fixed_token_ids)
            self._fixed_eos_logprobs = [-5.0 + 4.9 * i / max(n - 1, 1) for i in range(n)]

        if len(self._fixed_token_ids) != len(self._fixed_token_logprobs):
            raise ValueError(
                f"token_ids length ({len(self._fixed_token_ids)}) must match "
                f"token_logprobs length ({len(self._fixed_token_logprobs)})"
            )
        if len(self._fixed_token_ids) != len(self._fixed_eos_logprobs):
            raise ValueError(
                f"token_ids length ({len(self._fixed_token_ids)}) must match "
                f"eos_logprobs length ({len(self._fixed_eos_logprobs)})"
            )

        self._dummy = nn.Parameter(torch.zeros(1, device=device))

        self._last_eos_logprob_final: float | None = None

    def __call__(self, obs: str, temperature: float | None = None) -> PolicyOutput:
        action = {"type": "answer", "payload": "42"}

        logp: float | torch.Tensor
        ent: float | torch.Tensor
        if self.require_grad_logp:
            logp = torch.tensor(-0.5, device=self.device, requires_grad=True)
            ent = torch.tensor(0.1, device=self.device, requires_grad=True)
        else:
            logp = -0.5
            ent = 0.1

        self._last_eos_logprob_final = self._fixed_eos_logprobs[-1]

        return action, logp, ent

    def score_action(self, obs: str, action: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        logp = torch.tensor(-0.5, device=self.device, requires_grad=True)
        ent = torch.tensor(0.1, device=self.device, requires_grad=True)
        self._last_eos_logprob_final = self._fixed_eos_logprobs[-1]
        return logp, ent

    def parameters(self) -> Iterator[nn.Parameter]:
        yield self._dummy

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        yield ("dummy", self._dummy)

    def generate(self, messages: list[Message], *, gen: GenConfig) -> Generation:
        return Generation(
            text=self._fixed_text,
            token_ids=self._fixed_token_ids.copy(),
            token_logprobs=self._fixed_token_logprobs.copy(),
            finish_reason=self._fixed_finish_reason,
            eos_logprobs=self._fixed_eos_logprobs.copy(),
        )

    def logprobs(self, messages: list[Message], tokens: list[int]) -> TokenLogProbs:
        logprobs = [-0.1 * (i + 1) for i in range(len(tokens))]
        return TokenLogProbs(token_ids=tokens.copy(), logprobs=logprobs)

    def score_tokens(self, messages: list[Message], tokens: list[int]) -> torch.Tensor:
        if not tokens:
            return torch.tensor([], device=self.device, requires_grad=True)

        logprobs = torch.tensor(
            [-0.1 * (i + 1) for i in range(len(tokens))],
            device=self.device,
            requires_grad=True,
        )
        return logprobs
