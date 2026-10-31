"""Training must reject presets that the task runtime cannot execute."""

import pytest
from hydra import compose, initialize_config_module

from synthstats.cli.structured_config import register_train_schemas
from synthstats.cli.train import validate_train_config


def compose_train(overrides: list[str]):
    register_train_schemas()
    with initialize_config_module(config_module="synthstats.configs", version_base=None):
        return compose(config_name="train", overrides=overrides)


@pytest.mark.parametrize("judge", ["llm_critic_glm5", "hybrid_lm"])
def test_training_rejects_critic_without_container_support(judge):
    cfg = compose_train([f"judge={judge}"])

    with pytest.raises(ValueError, match="LLM critic.*task images"):
        validate_train_config(cfg)


@pytest.mark.parametrize("judge", [None, "likelihood", "composite"])
def test_training_accepts_supported_judges(judge):
    cfg = compose_train([] if judge is None else [f"judge={judge}"])

    validate_train_config(cfg)


@pytest.mark.parametrize("coefficient", [0.25, -0.25])
def test_training_rejects_unused_entropy_coefficients(coefficient):
    cfg = compose_train([f"objective.entropy_coef={coefficient}"])

    with pytest.raises(ValueError, match="objective.entropy_coef.*not supported"):
        validate_train_config(cfg)


@pytest.mark.parametrize("task", ["dugongs", "eight_schools", "peregrines", "surgical"])
def test_training_rejects_lazyppl_without_container_support(task, monkeypatch):
    monkeypatch.setenv("LAZYPPL_DEV_DIR", "/host-only/lazyppl")
    cfg = compose_train([f"env=lazyppl_{task}", "judge=likelihood"])

    with pytest.raises(ValueError, match="LazyPPL.*GHC.*Stack"):
        validate_train_config(cfg)
