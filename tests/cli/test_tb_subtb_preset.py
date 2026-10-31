"""A TB/SubTB blend must not silently select the pure TB objective."""

import pytest
from hydra import compose, initialize_config_module

from synthstats.cli.structured_config import register_train_schemas
from synthstats.cli.train import validate_train_config


@pytest.mark.parametrize("extra", [[], ["+objective.reward_floor=0.0001"]])
def test_training_rejects_unimplemented_tb_subtb_blend(extra):
    register_train_schemas()
    with initialize_config_module(config_module="synthstats.configs", version_base=None):
        cfg = compose(config_name="train", overrides=["objective=tb_subtb", *extra])

    with pytest.raises(ValueError, match="TB/SubTB blend.*not implemented"):
        validate_train_config(cfg)
