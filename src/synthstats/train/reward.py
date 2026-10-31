"""Reward-function adapter for judge-backed training integrations."""

from __future__ import annotations

from typing import Any

from synthstats.core.judge import Judge
from synthstats.core.judge_runtime import score_stubbed_messages
from synthstats.core.types import Reward


class SynthStatsRewardFunction:
    """Wrap a Judge with the scalar reward interface expected by training backends."""

    def __init__(self, judge: Judge, task_name: str) -> None:
        self.judge = judge
        self.task_name = task_name

    def __call__(
        self,
        messages: list[dict[str, str]],
        *,
        artifacts: dict[str, Any] | None = None,
    ) -> float:
        reward = score_stubbed_messages(
            judge=self.judge,
            task_name=self.task_name,
            messages=messages,
            artifacts=artifacts,
        )
        return reward.total

    def score_with_details(
        self,
        messages: list[dict[str, str]],
        *,
        artifacts: dict[str, Any] | None = None,
    ) -> Reward:
        return score_stubbed_messages(
            judge=self.judge,
            task_name=self.task_name,
            messages=messages,
            artifacts=artifacts,
        )


__all__ = ["SynthStatsRewardFunction"]
