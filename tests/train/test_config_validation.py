from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from synthstats.train.config_validation import (
    resolve_algorithm_config,
    validate_gfn_algorithm_config,
)


class TestConfigValidation:
    def test_resolve_algorithm_config_prefers_nested_trainer_algorithm(self):
        cfg = OmegaConf.create(
            {
                "trainer": {
                    "algorithm": {
                        "policy_loss_type": "gfn_tb",
                        "advantage_estimator": "tb_log_reward",
                    }
                }
            }
        )

        algorithm = resolve_algorithm_config(cfg)
        assert algorithm.policy_loss_type == "gfn_tb"
        assert algorithm.advantage_estimator == "tb_log_reward"

    def test_validate_gfn_algorithm_config_rejects_non_gfn_loss(self):
        cfg = OmegaConf.create(
            {
                "trainer": {
                    "algorithm": {
                        "policy_loss_type": "ppo",
                        "advantage_estimator": "tb_log_reward",
                    }
                }
            }
        )

        with pytest.raises(ValueError, match="policy_loss_type"):
            validate_gfn_algorithm_config(cfg)

    def test_validate_gfn_algorithm_config_rejects_non_tb_log_reward_estimator(self):
        cfg = OmegaConf.create(
            {
                "trainer": {
                    "algorithm": {
                        "policy_loss_type": "gfn_tb",
                        "advantage_estimator": "gae",
                    }
                }
            }
        )

        with pytest.raises(ValueError, match="advantage_estimator"):
            validate_gfn_algorithm_config(cfg)
