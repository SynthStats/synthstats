from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from synthstats.train.experiment_config import (
    as_positive_int,
    build_training_experiment_config,
    cfg_get,
    infer_policy_loss_type,
    promote_objective_algorithm_settings,
    to_plain_config,
    validate_training_experiment_config,
)


class TestExperimentConfig:
    def test_to_plain_config_resolves_omegaconf(self):
        cfg = OmegaConf.create({"objective": {"loss_type": "subtb"}})

        assert to_plain_config(cfg) == {"objective": {"loss_type": "subtb"}}

    def test_infer_policy_loss_type_maps_objective_loss(self):
        assert infer_policy_loss_type({"objective": {"loss_type": "subtb"}}, {}) == "gfn_subtb"
        assert infer_policy_loss_type({"objective": {"loss_type": "tb"}}, {}) == "gfn_tb"
        assert infer_policy_loss_type({"objective": {"loss_type": "ppo"}}, {}) is None

    def test_promote_objective_algorithm_settings_copies_lineage_fields(self):
        objective_cfg = {
            "subtb_lambda": 0.85,
            "tb_max_residual": 55.0,
            "lineage_weighting": {"metric": "n_with_verifier_reward"},
            "lineage_batching": {"metric": "n_with_verifier_reward"},
        }
        algorithm_cfg: dict[str, object] = {"policy_loss_type": "gfn_subtb"}

        promote_objective_algorithm_settings(objective_cfg, algorithm_cfg)

        assert algorithm_cfg["subtb_lambda"] == 0.85
        assert algorithm_cfg["tb_max_residual"] == 55.0
        assert algorithm_cfg["lineage_weighting"] == {"metric": "n_with_verifier_reward"}
        assert algorithm_cfg["lineage_batching"] == {"metric": "n_with_verifier_reward"}

    def test_cfg_get_walks_nested_dicts(self):
        cfg = {"trainer": {"algorithm": {"max_seq_len": 4096}}}

        assert cfg_get(cfg, "trainer", "algorithm", "max_seq_len") == 4096
        assert cfg_get(cfg, "trainer", "missing", default="fallback") == "fallback"

    def test_as_positive_int_rejects_bool_and_non_positive_values(self):
        assert as_positive_int(True) is None
        assert as_positive_int(0) is None
        assert as_positive_int(-3) is None
        assert as_positive_int(5) == 5

    def test_build_training_experiment_config_keeps_unset_parquet_inputs_empty(self):
        cfg = OmegaConf.create(
            {
                "backend": "vllm",
                "policy": {"model_name": "Qwen/Qwen3.5-4B"},
                "objective": {"loss_type": "subtb"},
                "runner": {
                    "trainer": {
                        "algorithm": {"max_seq_len": 4096},
                        "policy": {"model": {}},
                    },
                    "data": {"train_data": [], "val_data": []},
                    "generator": {"type": "harbor", "inference_engine": {}},
                },
            }
        )

        exp_cfg = build_training_experiment_config(cfg)

        assert exp_cfg.trainer.algorithm.policy_loss_type == "gfn_subtb"
        assert exp_cfg.trainer.algorithm.advantage_estimator == "tb_log_reward"
        assert exp_cfg.trainer.policy.model.path == "Qwen/Qwen3.5-4B"
        assert exp_cfg.generator.type == "harbor"
        assert exp_cfg.generator.store_all_messages is True
        assert exp_cfg.generator.enable_summarize is False
        assert exp_cfg.generator.inference_engine.backend == "vllm"
        assert exp_cfg.data.train_data == []

    def test_build_training_experiment_config_promotes_objective_settings(self):
        cfg = OmegaConf.create(
            {
                "backend": "vllm",
                "policy": {"model_name": "Qwen/Qwen3.5-4B"},
                "objective": {
                    "loss_type": "subtb",
                    "subtb_lambda": 0.85,
                    "tb_max_residual": 55.0,
                    "lineage_weighting": {"metric": "n_with_verifier_reward"},
                    "lineage_batching": {"metric": "n_with_verifier_reward"},
                },
                "runner": {
                    "trainer": {
                        "algorithm": {"max_seq_len": 4096},
                        "policy": {"model": {}},
                    },
                    "generator": {"type": "harbor", "inference_engine": {}},
                },
            }
        )

        exp_cfg = build_training_experiment_config(cfg)

        assert exp_cfg.trainer.algorithm.subtb_lambda == pytest.approx(0.85)
        assert exp_cfg.trainer.algorithm.tb_max_residual == pytest.approx(55.0)
        assert exp_cfg.trainer.algorithm.lineage_weighting.metric == "n_with_verifier_reward"
        assert exp_cfg.trainer.algorithm.lineage_batching.metric == "n_with_verifier_reward"

    def test_validate_training_experiment_config_requires_max_seq_len_for_harbor(self):
        cfg = OmegaConf.create(
            {
                "trainer": {
                    "algorithm": {
                        "policy_loss_type": "gfn_tb",
                        "advantage_estimator": "tb_log_reward",
                    }
                },
                "generator": {
                    "type": "harbor",
                    "inference_engine": {},
                },
            }
        )

        with pytest.raises(ValueError, match="max_seq_len"):
            validate_training_experiment_config(
                cfg,
                root_cfg={"policy": {"generation": {"max_new_tokens": 300}}},
            )

    def test_validate_training_experiment_config_rejects_colocation_mismatch(self):
        cfg = OmegaConf.create(
            {
                "trainer": {
                    "algorithm": {
                        "policy_loss_type": "gfn_tb",
                        "advantage_estimator": "tb_log_reward",
                        "max_seq_len": 4096,
                    },
                    "placement": {
                        "colocate_all": True,
                        "policy_num_nodes": 1,
                        "policy_num_gpus_per_node": 1,
                    },
                },
                "generator": {
                    "type": "harbor",
                    "inference_engine": {
                        "num_engines": 2,
                        "tensor_parallel_size": 1,
                    },
                },
            }
        )

        with pytest.raises(ValueError, match="colocate_all"):
            validate_training_experiment_config(
                cfg,
                root_cfg={"policy": {"generation": {"max_new_tokens": 300}}},
            )
