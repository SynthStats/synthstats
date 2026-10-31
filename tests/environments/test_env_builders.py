from pathlib import Path
from typing import Any, cast

import pytest
from omegaconf import OmegaConf

from synthstats.envs.builders import build_env, build_env_from_dict

CONFIG_DIR = Path(__file__).resolve().parents[2] / "src" / "synthstats" / "configs" / "env"


def test_build_env_requires_target() -> None:
    with pytest.raises(ValueError, match="cfg.env._target_ is required"):
        build_env(OmegaConf.create({"env": {}}))


def test_build_env_uses_env_judge_without_top_level_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = OmegaConf.create(
        {
            "env": {
                "_target_": "synthstats.envs.task_runtime_env.TaskRuntimeEnv",
                "judge": {
                    "_target_": "synthstats.judges.likelihood.LikelihoodJudge",
                    "beta": 0.5,
                },
            }
        }
    )
    captured: dict[str, object] = {}

    import hydra.utils

    def _fake_instantiate(env_cfg):
        captured["env_cfg"] = env_cfg
        return object()

    monkeypatch.setattr(hydra.utils, "instantiate", _fake_instantiate)

    _ = build_env(cfg)

    env_cfg = cast(dict[str, Any], captured["env_cfg"])
    assert env_cfg["judge"]["_target_"] == "synthstats.judges.likelihood.LikelihoodJudge"
    assert env_cfg["judge"]["beta"] == 0.5


def test_build_env_injects_top_level_judge_override(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = OmegaConf.create(
        {
            "env": {
                "_target_": "synthstats.envs.task_runtime_env.TaskRuntimeEnv",
                "judge": {
                    "_target_": "synthstats.judges.likelihood.LikelihoodJudge",
                    "beta": 1.0,
                },
            },
            "judge": {
                "_target_": "synthstats.judges.llm_critic.LLMCriticJudge",
                "model_name": "openai/gpt-5-mini",
            },
        }
    )
    captured: dict[str, object] = {}

    import hydra.utils

    def _fake_instantiate(env_cfg):
        captured["env_cfg"] = env_cfg
        return object()

    monkeypatch.setattr(hydra.utils, "instantiate", _fake_instantiate)

    _ = build_env(cfg)

    env_cfg = cast(dict[str, Any], captured["env_cfg"])
    assert env_cfg["judge"]["_target_"] == "synthstats.judges.llm_critic.LLMCriticJudge"
    assert env_cfg["judge"]["model_name"] == "openai/gpt-5-mini"
    # build_env should not mutate the base env config in-place
    assert cfg.env.judge._target_ == "synthstats.judges.likelihood.LikelihoodJudge"


# -- build_env_from_dict ----------------------------------------------------


def test_build_env_from_dict_flat_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flat dict with _target_ is wrapped under 'env' key automatically."""
    captured: dict[str, object] = {}

    import hydra.utils

    def _fake_instantiate(env_cfg):
        captured["env_cfg"] = env_cfg
        return object()

    monkeypatch.setattr(hydra.utils, "instantiate", _fake_instantiate)

    build_env_from_dict(
        {"_target_": "synthstats.envs.task_runtime_env.TaskRuntimeEnv", "name": "dugongs"}
    )

    env_cfg = cast(dict[str, Any], captured["env_cfg"])
    assert env_cfg["_target_"] == "synthstats.envs.task_runtime_env.TaskRuntimeEnv"


def test_build_env_from_dict_nested_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dict with explicit 'env' key is used as-is."""
    captured: dict[str, object] = {}

    import hydra.utils

    def _fake_instantiate(env_cfg):
        captured["env_cfg"] = env_cfg
        return object()

    monkeypatch.setattr(hydra.utils, "instantiate", _fake_instantiate)

    build_env_from_dict({"env": {"_target_": "synthstats.envs.task_runtime_env.TaskRuntimeEnv"}})

    env_cfg = cast(dict[str, Any], captured["env_cfg"])
    assert env_cfg["_target_"] == "synthstats.envs.task_runtime_env.TaskRuntimeEnv"


def test_build_env_from_dict_accepts_task_runtime_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    import hydra.utils

    def _fake_instantiate(env_cfg):
        captured["env_cfg"] = env_cfg
        return object()

    monkeypatch.setattr(hydra.utils, "instantiate", _fake_instantiate)

    build_env_from_dict(
        {
            "_target_": "synthstats.envs.task_runtime_env.TaskRuntimeEnv",
            "name": "dugongs",
        }
    )

    env_cfg = cast(dict[str, Any], captured["env_cfg"])
    assert env_cfg["_target_"] == "synthstats.envs.task_runtime_env.TaskRuntimeEnv"


def test_build_env_from_dict_missing_target() -> None:
    """Dict without _target_ raises ValueError."""
    with pytest.raises(ValueError, match="_target_"):
        build_env_from_dict({"name": "dugongs"})


def test_statistical_modeling_base_env_config_uses_strict_loo_gate() -> None:
    cfg = OmegaConf.load(CONFIG_DIR / "_statistical_modeling_base.yaml")

    assert cfg._target_ == "synthstats.envs.task_runtime_env.TaskRuntimeEnv"
    assert cfg.config._target_ == "synthstats.envs.task_runtime_env.TaskRuntimeEnvConfig"
    assert cfg.task._target_ == "synthstats.tasks.statistical_modeling.task.StatisticalModelingTask"
    assert cfg.judge._target_ == "synthstats.judges.likelihood.LikelihoodJudge"
    assert cfg.judge.loo_gate_mode == "strict"


def test_lazyppl_base_env_config_uses_strict_loo_gate() -> None:
    cfg = OmegaConf.load(CONFIG_DIR / "_lazyppl_base.yaml")

    assert cfg._target_ == "synthstats.envs.task_runtime_env.TaskRuntimeEnv"
    assert cfg.config._target_ == "synthstats.envs.task_runtime_env.TaskRuntimeEnvConfig"
    assert cfg.task._target_ == "synthstats.tasks.statistical_modeling.task.StatisticalModelingTask"
    likelihood_cfg = cfg.judge.judges[1].judge
    assert likelihood_cfg._target_ == "synthstats.judges.likelihood.LikelihoodJudge"
    assert likelihood_cfg.loo_gate_mode == "strict"


def test_build_env_supports_shorthand_composite_judge_override() -> None:
    base_env = OmegaConf.load(CONFIG_DIR / "dugongs.yaml")
    base_base = OmegaConf.load(CONFIG_DIR / "_statistical_modeling_base.yaml")
    merged_env = OmegaConf.merge(base_base, base_env)
    if "defaults" in merged_env:
        del merged_env["defaults"]
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "env": merged_env,
            "judge": {
                "_target_": None,
                "scalarization_mode": None,
                "beta": None,
                "loo_gate_mode": None,
                "config": {},
                "judges": [
                    {"type": "likelihood", "weight": 1.0},
                    {"type": "formatting", "weight": 0.1},
                ],
                "type": None,
                "weight": None,
            },
        }
    )
    env = build_env(cfg)
    assert type(env.judge).__name__ == "CompositeJudge"
    assert [type(j).__name__ for j, _ in env.judge.judges] == [
        "LikelihoodJudge",
        "FormattingJudge",
    ]
    assert [w for _, w in env.judge.judges] == [pytest.approx(1.0), pytest.approx(0.1)]
