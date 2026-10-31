"""Judge protocol."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from synthstats.core.types import Reward, Trajectory


@runtime_checkable
class Judge(Protocol):
    """Reward computation interface."""

    def score(
        self, *, task_name: str, trajectory: Trajectory, artifacts: dict[str, Any]
    ) -> Reward: ...
