"""Gym-environment registration helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

ENVIRONMENT_REGISTRATIONS: tuple[tuple[str, str], ...] = (
    ("synthstats-task-runtime-v0", "synthstats.envs.task_runtime_env:TaskRuntimeEnv"),
)


def register_environments(
    gym_register: Callable[..., None] | None = None,
) -> None:
    """Register env IDs with gymnasium."""
    try:
        if gym_register is None:
            from gymnasium import register as imported_gym_register  # noqa: PLC0415

            register_callable = imported_gym_register
        else:
            register_callable = gym_register

        for env_id, entry_point in ENVIRONMENT_REGISTRATIONS:
            register_callable(id=env_id, entry_point=entry_point)
            logger.info("Registered %s with gymnasium", env_id)
    except ImportError as exc:
        logger.debug("Could not register SynthStats environments: %s", exc)


__all__ = ["ENVIRONMENT_REGISTRATIONS", "register_environments"]
