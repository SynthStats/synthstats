"""Typed configuration dataclasses for the Harbor runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from synthstats.core.config import (
    BaseRunnerConfig,
    parse_base_runner_fields,
)
from synthstats.core.config import (
    config_get as _get,
)


@dataclass
class HarborConfig:
    """Harbor-specific collection settings."""

    trajectories_dir: str | None = None
    rescore_with_policy: bool = True
    compute_eos_logprobs: bool = True


@dataclass
class HarborRunnerConfig(BaseRunnerConfig):
    harbor: HarborConfig = field(default_factory=HarborConfig)


def parse_harbor_runner_config(cfg: Any) -> HarborRunnerConfig:
    harbor_raw = _get(cfg, "runner", "harbor", default={})

    trajectories_dir = _get(harbor_raw, "trajectories_dir")
    if trajectories_dir is None:
        raise ValueError(
            "runner.harbor.trajectories_dir is required. "
            "Point it at a directory of ATIF trajectory JSON files."
        )

    base_fields = parse_base_runner_fields(cfg)

    return HarborRunnerConfig(
        **base_fields,
        harbor=HarborConfig(
            trajectories_dir=trajectories_dir,
            rescore_with_policy=_get(harbor_raw, "rescore_with_policy", default=True),
            compute_eos_logprobs=_get(harbor_raw, "compute_eos_logprobs", default=True),
        ),
    )
