"""Base protocol for checkpoint managers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """Complete training state for checkpoint/resume.

    Unified schema for training checkpoints. Optional fields default to
    None/empty so distributed runners (SkyRL) can omit what they don't use.
    """

    step_count: int
    logZ: float
    model_state_dict: dict[str, Any] | None
    optimizer_state_dict: dict[str, Any] | None
    rng_states: dict[str, Any]
    replay_buffer: dict[str, Any] | None
    config: dict[str, Any]
    learner_state: dict[str, Any] | None = None
    scheduler_state_dict: dict[str, Any] | None = None
    metrics_history: list[dict[str, float]] = field(default_factory=list)
    # cross-runner fields
    gfn_state: dict[str, Any] | None = None
    runner_type: str = "unknown"
    checkpoint_version: int = 3

    def to_dict(self) -> dict[str, Any]:
        d = {
            "step_count": self.step_count,
            "logZ": self.logZ,
            "model_state_dict": self.model_state_dict,
            "optimizer_state_dict": self.optimizer_state_dict,
            "rng_states": self.rng_states,
            "replay_buffer": self.replay_buffer,
            "config": self.config,
            "learner_state": self.learner_state,
            "scheduler_state_dict": self.scheduler_state_dict,
            "metrics_history": self.metrics_history,
            "runner_type": self.runner_type,
            "checkpoint_version": self.checkpoint_version,
        }
        if self.gfn_state is not None:
            d["gfn_state"] = self.gfn_state
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointState:
        return cls(
            step_count=data["step_count"],
            logZ=data["logZ"],
            model_state_dict=data.get("model_state_dict"),
            optimizer_state_dict=data.get("optimizer_state_dict"),
            rng_states=data.get("rng_states", {}),
            replay_buffer=data.get("replay_buffer"),
            config=data.get("config", {}),
            learner_state=data.get("learner_state"),
            scheduler_state_dict=data.get("scheduler_state_dict"),
            metrics_history=data.get("metrics_history", []),
            gfn_state=data.get("gfn_state"),
            runner_type=data.get("runner_type", "unknown"),
            checkpoint_version=data.get("checkpoint_version", 1),
        )

    @classmethod
    def from_gfn_state(
        cls,
        gfn_state: dict[str, Any],
        step_count: int | None = None,
    ) -> CheckpointState:
        """Create a CheckpointState from a SkyRL GFlowNetTrainer state dict.

        Maps {logZ, logZ_optimizer, gfn_step_count} into the unified schema.
        """
        return cls(
            step_count=step_count or gfn_state.get("gfn_step_count", 0),
            logZ=float(gfn_state.get("logZ", 0.0)),
            model_state_dict=None,
            optimizer_state_dict=None,
            rng_states={},
            replay_buffer=None,
            config={},
            gfn_state=gfn_state,
            runner_type="skyrl",
            checkpoint_version=3,
        )

    def to_gfn_state(self) -> dict[str, Any]:
        """Extract GFlowNetTrainer-compatible state dict.

        If gfn_state was stored, return it directly. Otherwise reconstruct
        from the base fields (logZ + step_count, no optimizer state).
        """
        if self.gfn_state is not None:
            return self.gfn_state
        return {
            "logZ": self.logZ,
            "gfn_step_count": self.step_count,
        }


@runtime_checkable
class CheckpointManager(Protocol):
    """Protocol for checkpoint management."""

    def maybe_save(
        self,
        step: int,
        learner: Any,
        policy: Any | None = None,
        replay_buffer: Any | None = None,
        metrics_history: list[dict[str, float]] | None = None,
    ) -> Path | None: ...

    def save(
        self,
        step: int,
        learner: Any,
        policy: Any | None = None,
        replay_buffer: Any | None = None,
        metrics_history: list[dict[str, float]] | None = None,
    ) -> Path: ...

    def load(self, path: str | Path) -> CheckpointState: ...

    def find_latest(self) -> Path | None: ...

    def restore(
        self,
        learner: Any,
        policy: Any | None = None,
        replay_buffer: Any | None = None,
        path: str | Path | None = None,
    ) -> int: ...


def cleanup_old_checkpoints(
    checkpoint_dir: Path,
    keep_last_n: int,
) -> list[Path]:
    """Remove old step directories, keeping the most recent ``keep_last_n``."""
    import re
    import shutil

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []

    step_re = re.compile(r"^step_(\d+)$")
    step_dirs: list[tuple[int, Path]] = []
    for d in checkpoint_dir.iterdir():
        if d.is_dir():
            m = step_re.match(d.name)
            if m:
                step_dirs.append((int(m.group(1)), d))

    step_dirs.sort()

    if keep_last_n <= 0 or len(step_dirs) <= keep_last_n:
        return []

    to_remove = step_dirs[:-keep_last_n]
    removed: list[Path] = []

    for _step, d in to_remove:
        try:
            shutil.rmtree(d)
            removed.append(d)
            logger.debug("removed old checkpoint: %s", d)
        except OSError as e:
            logger.warning("failed to remove checkpoint %s: %s", d, e)

    if removed:
        logger.info("cleaned up %d old checkpoints", len(removed))

    return removed
