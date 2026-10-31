"""Task protocol."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from synthstats.core.types import Action, Message, StepResult


@runtime_checkable
class Task(Protocol):
    """Domain plugin interface."""

    name: str

    def reset(self, seed: int | None = None) -> Any: ...

    def observe(self, state: Any) -> list[Message]: ...

    def step(self, state: Any, action: Action) -> StepResult: ...

    def native_actions(self) -> frozenset[str]:
        """Action names handled by task.step(), not by executors.

        Override to declare task-owned actions (e.g. {"query"} for statistical modeling).
        Rollout loops skip executor dispatch for these names.
        """
        return frozenset()
