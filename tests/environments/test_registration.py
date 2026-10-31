from synthstats.envs.registration import (
    ENVIRONMENT_REGISTRATIONS,
    register_environments,
)


def test_environment_registrations_expose_env_ids() -> None:
    assert ENVIRONMENT_REGISTRATIONS == (
        ("synthstats-task-runtime-v0", "synthstats.envs.task_runtime_env:TaskRuntimeEnv"),
    )


def test_register_environments_uses_supplied_register_callable() -> None:
    registrations: list[dict[str, str]] = []

    def _fake_register(*, id: str, entry_point: str) -> None:
        registrations.append({"id": id, "entry_point": entry_point})

    register_environments(_fake_register)

    assert registrations == [
        {
            "id": "synthstats-task-runtime-v0",
            "entry_point": "synthstats.envs.task_runtime_env:TaskRuntimeEnv",
        },
    ]
