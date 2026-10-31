"""Base protocol for training runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class RunResult:
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoints: list[str] = field(default_factory=list)
    interrupted: bool = False
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


@runtime_checkable
class Runner(Protocol):
    def run(self) -> RunResult: ...
