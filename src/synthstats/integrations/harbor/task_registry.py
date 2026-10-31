"""Local Harbor task-package registry for SynthStats benchmarks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from synthstats.envs.builders import normalize_judge_config

TASK_PACKAGES_DIR = Path(__file__).resolve().parent / "task_packages"
REGISTERED_TASK_PACKAGES = (
    "dugongs",
    "peregrines",
    "eight_schools",
    "surgical",
)


@dataclass(frozen=True)
class HarborTaskRequest:
    """A Harbor task package plus the SynthStats env payload it should run."""

    task_name: str
    task_path: str
    task_config: dict[str, Any]
    source: str | None = None
    git_url: str | None = None
    git_commit_id: str | None = None

    def to_env_extra(self) -> dict[str, Any]:
        payload = {
            "task_name": self.task_name,
            "task_path": self.task_path,
            "task_config": deepcopy(self.task_config),
        }
        if self.source is not None:
            payload["source"] = self.source
        if self.git_url is not None:
            payload["git_url"] = self.git_url
        if self.git_commit_id is not None:
            payload["git_commit_id"] = self.git_commit_id
        return payload


def _to_plain(value: Any) -> Any:
    if value is None:
        return None
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return deepcopy(value)


def _pick_packaged_task_name(*, env_name: Any, task_env_name: Any) -> str | None:
    if isinstance(env_name, str) and env_name in REGISTERED_TASK_PACKAGES:
        return env_name
    if isinstance(task_env_name, str) and task_env_name in REGISTERED_TASK_PACKAGES:
        return task_env_name
    return None


def _has_meaningful_judge_override(judge_cfg: dict[str, Any]) -> bool:
    return bool(normalize_judge_config(judge_cfg))


def resolve_task_package_path(task_name: str) -> Path:
    if task_name not in REGISTERED_TASK_PACKAGES:
        available = ", ".join(sorted(REGISTERED_TASK_PACKAGES))
        raise ValueError(
            f"No local Harbor task package is registered for task '{task_name}'. "
            f"Available packages: {available}"
        )
    package_path = TASK_PACKAGES_DIR / task_name
    if not package_path.exists():
        raise ValueError(
            f"Harbor task package '{task_name}' is registered but missing on disk. "
            f"Expected to find: {package_path}"
        )
    return package_path


def infer_task_name(cfg: Any) -> str:
    env_cfg = _to_plain(getattr(cfg, "get", lambda *_args, **_kwargs: None)("env"))
    if not isinstance(env_cfg, dict):
        raise ValueError("cfg.env is required to build a Harbor task request")

    env_name = env_cfg.get("name")
    task_cfg = env_cfg.get("task", {})
    task_env_name = task_cfg.get("env_name") if isinstance(task_cfg, dict) else None

    packaged_task_name = _pick_packaged_task_name(
        env_name=env_name,
        task_env_name=task_env_name,
    )
    if packaged_task_name is not None:
        return packaged_task_name

    if isinstance(env_name, str) and env_name:
        return env_name

    if isinstance(task_cfg, dict):
        if isinstance(task_env_name, str) and task_env_name:
            return task_env_name

    raise ValueError("cfg.env.name or cfg.env.task.env_name is required")


def build_task_config_payload(cfg: Any) -> dict[str, Any]:
    env_cfg = _to_plain(getattr(cfg, "get", lambda *_args, **_kwargs: None)("env"))
    if not isinstance(env_cfg, dict):
        raise ValueError("cfg.env is required to build a Harbor task request")

    seed = _to_plain(getattr(cfg, "get", lambda *_args, **_kwargs: None)("seed"))
    if isinstance(seed, int) and not isinstance(seed, bool):
        env_cfg = deepcopy(env_cfg)
        if isinstance(env_cfg.get("config"), dict):
            env_cfg["config"] = dict(env_cfg["config"])
            env_cfg["config"]["evaluation_seed"] = seed

    payload: dict[str, Any] = {"env": env_cfg}
    if isinstance(seed, int) and not isinstance(seed, bool):
        payload["seed"] = seed
    judge_cfg = _to_plain(getattr(cfg, "get", lambda *_args, **_kwargs: None)("judge"))
    if isinstance(judge_cfg, dict):
        normalized_judge = normalize_judge_config(judge_cfg)
        if normalized_judge:
            payload["judge"] = normalized_judge
    return payload


def build_local_task_request(cfg: Any) -> dict[str, Any]:
    task_name = infer_task_name(cfg)
    request = HarborTaskRequest(
        task_name=task_name,
        task_path=str(resolve_task_package_path(task_name)),
        task_config=build_task_config_payload(cfg),
        source="local_task_package",
    )
    return request.to_env_extra()


__all__ = [
    "HarborTaskRequest",
    "REGISTERED_TASK_PACKAGES",
    "TASK_PACKAGES_DIR",
    "build_local_task_request",
    "build_task_config_payload",
    "infer_task_name",
    "resolve_task_package_path",
]
