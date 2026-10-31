"""Reward temperature scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class RewardTemperatureSchedule:
    """Anneals reward temperature from start to end over horizon steps."""

    start: float = 1.0
    end: float = 0.1
    horizon: int = 1000
    mode: Literal["linear", "cosine"] = "linear"

    def get(self, step: int) -> float:
        if self.horizon <= 0:
            return self.end

        t = min(1.0, step / self.horizon)

        if self.mode == "cosine":
            import math

            t = 0.5 * (1 - math.cos(math.pi * t))

        return self.start + (self.end - self.start) * t

    def scale_reward(self, log_reward: float, step: int) -> float:
        temp = self.get(step)
        return log_reward / temp if temp > 0 else log_reward
