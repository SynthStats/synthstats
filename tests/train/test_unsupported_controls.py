"""Unsupported training controls fail before configuration can discard them."""

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from synthstats.cli.structured_config import register_train_schemas
from synthstats.cli.train import validate_train_config
from synthstats.train.experiment_config import (
    build_training_experiment_config,
    validate_training_experiment_config,
)

ENTRY_POINTS = (
    "cli",
    "builder",
    "validator",
)


@pytest.fixture
def config():
    register_train_schemas()
    with initialize_config_module(config_module="synthstats.configs", version_base=None):
        cfg = compose(config_name="train")
    return OmegaConf.to_container(cfg, resolve=True)


def check_entry_point(entry_point, config):
    cfg = OmegaConf.create(config)
    if entry_point == "cli":
        validate_train_config(cfg)
    elif entry_point == "builder":
        build_training_experiment_config(cfg)
    elif entry_point == "validator":
        validate_training_experiment_config(cfg.runner, root_cfg=config)
    else:
        raise AssertionError(f"Unknown entry point: {entry_point}")


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
@pytest.mark.parametrize("coefficient", [0.25, -0.25, False, True])
def test_training_rejects_unused_or_boolean_entropy_coefficients(entry_point, coefficient, config):
    config["objective"]["entropy_coef"] = coefficient

    with pytest.raises(ValueError, match="objective.entropy_coef.*not supported"):
        check_entry_point(entry_point, config)


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
@pytest.mark.parametrize("coefficient", [None, 0, 0.0])
def test_training_accepts_omitted_or_numeric_zero_entropy(entry_point, coefficient, config):
    if coefficient is None:
        config["objective"].pop("entropy_coef", None)
    else:
        config["objective"]["entropy_coef"] = coefficient

    check_entry_point(entry_point, config)


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
@pytest.mark.parametrize(
    "target",
    ["synthstats.policies.runpod_policy.RunPodPolicy", "some.module.Policy"],
)
def test_training_rejects_a_policy_target_instead_of_ignoring_it(entry_point, target, config):
    config["policy"]["_target_"] = target

    with pytest.raises(ValueError, match=r"policy\._target_ is not supported"):
        check_entry_point(entry_point, config)
