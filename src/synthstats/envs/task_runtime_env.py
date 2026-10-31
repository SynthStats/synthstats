"""Generic task runtime environment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from synthstats.core.constants import PYMC_EXECUTOR_TIMEOUT_S
from synthstats.envs.task_runtime_base import (
    SKYRL_AVAILABLE,
    TaskRuntimeConfig,
    TaskRuntimeEnvBase,
)
from synthstats.executors.elpd import ELPDRunner

if TYPE_CHECKING:
    from synthstats.core.executor import Executor
    from synthstats.core.judge import Judge
    from synthstats.core.task import Task
    from synthstats.runtime.codecs import ActionCodec


class TaskRuntimeEnvConfig(TaskRuntimeConfig):
    """Config for the generic task runtime environment."""


class TaskRuntimeEnv(TaskRuntimeEnvBase):
    """Wrapper around the generic task runtime implementation."""

    def __init__(
        self,
        task: Task,
        codec: ActionCodec,
        executors: dict[str, Executor] | None = None,
        judge: Judge | None = None,
        config: TaskRuntimeEnvConfig | None = None,
        *,
        name: str | None = None,
        max_steps: int | None = None,
        worker_pool_size: int | None = None,
    ) -> None:
        super().__init__(
            task=task,
            codec=codec,
            executors=executors,
            judge=judge,
            config=config or TaskRuntimeEnvConfig(),
            name=name,
            max_steps=max_steps,
            worker_pool_size=worker_pool_size,
        )

    def _build_elpd_runner(self) -> Any:
        return ELPDRunner(
            timeout=PYMC_EXECUTOR_TIMEOUT_S,
            random_seed=getattr(self, "_random_seed", None),
            pool=self._get_pool(),
        )


__all__ = ["SKYRL_AVAILABLE", "TaskRuntimeEnv", "TaskRuntimeEnvConfig"]
