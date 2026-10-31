from __future__ import annotations

from types import SimpleNamespace

from synthstats.envs.task_runtime_base import TaskRuntimeConfig, TaskRuntimeEnvBase
from synthstats.envs.task_runtime_env import TaskRuntimeEnv, TaskRuntimeEnvConfig


def test_task_runtime_env_exports() -> None:
    assert issubclass(TaskRuntimeEnv, TaskRuntimeEnvBase)
    assert issubclass(TaskRuntimeEnvConfig, TaskRuntimeConfig)
    assert TaskRuntimeEnv.__name__ == "TaskRuntimeEnv"
    assert TaskRuntimeEnvConfig.__name__ == "TaskRuntimeEnvConfig"


def test_task_runtime_env_uses_default_config() -> None:
    env = TaskRuntimeEnv(
        task=SimpleNamespace(name="dummy"),
        codec=SimpleNamespace(),
        executors={},
        judge=None,
    )

    assert isinstance(env, TaskRuntimeEnv)
    assert isinstance(env, TaskRuntimeEnvBase)
    assert isinstance(env.config, TaskRuntimeEnvConfig)


def test_env_package_exports_runtime() -> None:
    import synthstats.envs as envs_pkg

    assert envs_pkg.TaskRuntimeEnv is TaskRuntimeEnv
    assert envs_pkg.TaskRuntimeEnvConfig is TaskRuntimeEnvConfig
    assert envs_pkg.__all__ == [
        "TaskRuntimeEnv",
        "TaskRuntimeEnvConfig",
        "SKYRL_AVAILABLE",
    ]
