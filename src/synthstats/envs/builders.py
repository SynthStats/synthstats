"""Environment builders for training runners.

Preserves dependency inversion: train/ never imports tasks/ directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf

if TYPE_CHECKING:
    from synthstats.envs.task_runtime_env import TaskRuntimeEnv

_JUDGE_TYPE_TARGETS: dict[str, str] = {
    "likelihood": "synthstats.judges.likelihood.LikelihoodJudge",
    "formatting": "synthstats.judges.formatting.FormattingJudge",
    "compilation": "synthstats.judges.compilation.CompilationJudge",
    "llm_critic": "synthstats.judges.llm_critic.LLMCriticJudge",
    "composite": "synthstats.judges.composite.CompositeJudge",
}

_FOREIGN_SURFACE_KEYS_BY_TARGET: dict[str, frozenset[str]] = {
    "synthstats.judges.likelihood.LikelihoodJudge": frozenset(
        {"scalarization_mode", "config", "judges", "type", "weight"}
    ),
    "synthstats.judges.composite.CompositeJudge": frozenset(
        {"beta", "loo_gate_mode", "config", "type", "weight"}
    ),
    "synthstats.judges.llm_critic.LLMCriticJudge": frozenset(
        {"scalarization_mode", "beta", "loo_gate_mode", "judges", "type", "weight"}
    ),
    "synthstats.judges.formatting.FormattingJudge": frozenset(
        {"scalarization_mode", "beta", "loo_gate_mode", "config", "judges", "type", "weight"}
    ),
    "synthstats.judges.compilation.CompilationJudge": frozenset(
        {"scalarization_mode", "beta", "loo_gate_mode", "config", "judges", "type", "weight"}
    ),
}


def normalize_judge_config(judge_cfg: Any) -> dict[str, Any]:
    """Normalize a judge config mapping and prune foreign JudgeSurfaceConfig defaults."""
    if judge_cfg is None:
        return {}
    if OmegaConf.is_config(judge_cfg):
        raw_plain = OmegaConf.to_container(judge_cfg, resolve=True)
        if not isinstance(raw_plain, Mapping):
            return {}
        plain: dict[str, Any] = {str(k): v for k, v in raw_plain.items()}
    elif isinstance(judge_cfg, Mapping):
        plain = {str(k): deepcopy(v) for k, v in judge_cfg.items()}
    else:
        return {}

    raw_target = plain.get("_target_")
    target: str | None = raw_target if isinstance(raw_target, str) and raw_target.strip() else None
    raw_type = plain.get("type")
    judge_type: str | None = raw_type if isinstance(raw_type, str) and raw_type.strip() else None
    raw_judges = plain.get("judges")

    if target is None:
        if judge_type is not None and judge_type in _JUDGE_TYPE_TARGETS:
            target = _JUDGE_TYPE_TARGETS[judge_type]
        elif isinstance(raw_judges, list) and len(raw_judges) > 0:
            target = _JUDGE_TYPE_TARGETS["composite"]

    foreign_keys = _FOREIGN_SURFACE_KEYS_BY_TARGET.get(target or "", frozenset())
    cleaned: dict[str, Any] = {}
    for key, value in plain.items():
        if key in ("_target_", "type"):
            continue
        if value is None:
            continue
        if key in foreign_keys:
            continue
        if key == "config" and isinstance(value, Mapping) and not value:
            continue
        if key == "judges" and isinstance(value, list):
            if not value and target != "synthstats.judges.composite.CompositeJudge":
                continue
            normalized_judges: list[Any] = []
            for entry in value:
                if isinstance(entry, Mapping):
                    entry_dict = dict(entry)
                    if "judge" in entry_dict:
                        sub_judge = entry_dict["judge"]
                        if isinstance(sub_judge, Mapping) or OmegaConf.is_config(sub_judge):
                            entry_dict["judge"] = normalize_judge_config(sub_judge)
                        normalized_judges.append(entry_dict)
                    else:
                        weight = entry_dict.pop("weight", 1.0)
                        sub_judge = normalize_judge_config(entry_dict)
                        normalized_judges.append(
                            {
                                "judge": sub_judge,
                                "weight": float(weight) if weight is not None else 1.0,
                            }
                        )
                else:
                    normalized_judges.append(deepcopy(entry))
            cleaned["judges"] = normalized_judges
            continue
        cleaned[key] = deepcopy(value)

    if target is not None:
        return {"_target_": target, **cleaned}
    if judge_type is not None:
        cleaned["type"] = judge_type
    return cleaned


def merge_judge_configs(
    base_judge: Any,
    override_judge: Any,
) -> dict[str, Any]:
    """Merge an override judge mapping into a base judge mapping."""
    norm_base = normalize_judge_config(base_judge)
    norm_override = normalize_judge_config(override_judge)
    if not norm_override:
        return norm_base
    if not norm_base:
        return norm_override

    base_target = norm_base.get("_target_")
    override_target = norm_override.get("_target_")
    if override_target is not None and base_target is not None and override_target != base_target:
        return norm_override

    merged = deepcopy(norm_base)
    for key, value in norm_override.items():
        if (
            key == "config"
            and isinstance(merged.get("config"), Mapping)
            and isinstance(value, Mapping)
        ):
            merged["config"] = {**dict(merged["config"]), **dict(value)}
        else:
            merged[key] = deepcopy(value)
    return normalize_judge_config(merged)


def build_env_from_dict(config: dict[str, Any]) -> TaskRuntimeEnv:
    """Build env from a plain dict (Harbor Docker agent path).

    Accepts either {"_target_": "...", ...} (flat env config) or
    {"env": {"_target_": "...", ...}} (full config with env key).
    """
    if "env" not in config and "_target_" in config:
        config = {"env": config}

    cfg = OmegaConf.create(config)
    return build_env(cfg)


def build_env(cfg: Any) -> TaskRuntimeEnv:
    """Build an environment from config. Requires cfg.env._target_."""
    env_cfg = deepcopy(cfg.get("env", {}) if hasattr(cfg, "get") else {})
    target = env_cfg.get("_target_", None) if hasattr(env_cfg, "get") else None
    if target is None:
        raise ValueError("cfg.env._target_ is required")

    env_judge_raw = env_cfg.get("judge", None) if hasattr(env_cfg, "get") else None
    judge_cfg = cfg.get("judge", None) if hasattr(cfg, "get") else None
    merged_judge = merge_judge_configs(env_judge_raw, judge_cfg)
    if merged_judge.get("_target_") is not None:
        env_cfg["judge"] = merged_judge

    seed = cfg.get("seed", None) if hasattr(cfg, "get") else None
    if isinstance(seed, int) and not isinstance(seed, bool):
        runtime_cfg = env_cfg.get("config", None) if hasattr(env_cfg, "get") else None
        if runtime_cfg is not None and hasattr(runtime_cfg, "__setitem__"):
            runtime_cfg["evaluation_seed"] = seed

    from hydra.utils import instantiate

    return instantiate(env_cfg)
