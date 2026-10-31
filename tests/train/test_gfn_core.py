from __future__ import annotations

import pytest
from omegaconf import DictConfig

from synthstats.train.gfn_core import GFlowNetTrainerCore, LogZModule, resolve_logz_params


class _CoreTrainer(GFlowNetTrainerCore):
    pass


class TestLogZModule:
    def test_forward(self) -> None:
        module = LogZModule(init_value=1.5)
        assert module().item() == pytest.approx(1.5)
        assert module.logZ.requires_grad

    def test_gradient_flows(self) -> None:
        module = LogZModule(0.0)
        loss = (module() - 5.0) ** 2
        loss.backward()
        assert module.logZ.grad is not None


class TestGFlowNetTrainerCore:
    def test_state_dict_roundtrip(self) -> None:
        trainer = _CoreTrainer()
        trainer.init_gfn(logZ_init=3.0, logZ_lr=0.1)
        state = trainer.gfn_state_dict()
        assert state["logZ"] == pytest.approx(3.0)
        assert state["gfn_step_count"] == 0

        trainer2 = _CoreTrainer()
        trainer2.init_gfn(logZ_init=0.0, logZ_lr=0.1)
        trainer2.gfn_load_state_dict(state)
        assert trainer2.logZ.item() == pytest.approx(3.0)
        assert trainer2._gfn_step_count == 0

    def test_optimizer_step_updates_logz_and_step_count(self) -> None:
        trainer = _CoreTrainer()
        trainer.init_gfn(logZ_init=1.0, logZ_lr=0.1)

        loss = (trainer.logZ - 5.0) ** 2
        loss.backward()
        metrics = trainer.gfn_optimizer_step()

        assert metrics["gfn_step"] == 1
        assert metrics["logZ"] != pytest.approx(1.0)

    def test_inject_logz_into_dict_config(self) -> None:
        trainer = _CoreTrainer()
        trainer.init_gfn(logZ_init=4.0)

        config: dict[str, object] = {"algorithm": {}}
        trainer.inject_logZ_into_config(config)

        algorithm = config["algorithm"]
        assert isinstance(algorithm, dict)
        assert algorithm["logZ"] == pytest.approx(4.0)

    def test_inject_logz_into_omegaconf(self) -> None:
        trainer = _CoreTrainer()
        trainer.init_gfn(logZ_init=2.5)

        config = DictConfig({"algorithm": {"logZ": 0.0}})
        trainer.inject_logZ_into_config(config)

        assert config.algorithm.logZ == 2.5
        assert hasattr(config, "logZ_module")

    def test_checkpoint_roundtrip(self) -> None:
        trainer = _CoreTrainer()
        trainer.init_gfn(logZ_init=2.0)

        checkpoint = trainer.to_checkpoint_state()

        trainer2 = _CoreTrainer()
        trainer2.init_gfn(logZ_init=0.0)
        trainer2.load_from_checkpoint_state(checkpoint)

        assert trainer2.logZ.item() == pytest.approx(2.0)

    def test_merge_rollout_metrics_prefixes_numeric_values(self) -> None:
        trainer = _CoreTrainer()
        trainer.init_gfn()

        class FakeBatch:
            rollout_metrics = {
                "n_success": 2,
                "n_with_trace_id": 1,
                "reward_sources": ["verifier"],
                "reward_evidence_kinds": ["verifier", "heuristic"],
                "reward_truth_owners": ["verifier", "heuristic"],
            }

        assert trainer.merge_rollout_metrics(FakeBatch()) == {
            "rollout_n_success": 2.0,
            "rollout_n_with_trace_id": 1.0,
            "rollout_n_reward_evidence_kind_heuristic": 1.0,
            "rollout_n_reward_evidence_kind_verifier": 1.0,
            "rollout_reward_evidence_kind_heuristic_fraction": pytest.approx(0.5),
            "rollout_reward_evidence_kind_verifier_fraction": pytest.approx(0.5),
            "rollout_n_reward_truth_owner_heuristic": 1.0,
            "rollout_n_reward_truth_owner_verifier": 1.0,
            "rollout_reward_truth_owner_heuristic_fraction": pytest.approx(0.5),
            "rollout_reward_truth_owner_verifier_fraction": pytest.approx(0.5),
        }

    def test_inject_lineage_loss_weight_into_dict_algorithm_config(self) -> None:
        trainer = _CoreTrainer()
        trainer.init_gfn()

        config: dict[str, object] = {
            "algorithm": {
                "lineage_weighting": {
                    "metric": "n_with_verifier_reward",
                    "denominator_metric": "n_success",
                    "base_weight": 1.0,
                    "scale": 0.5,
                    "min_weight": 1.0,
                    "max_weight": 2.0,
                }
            }
        }

        class FakeBatch:
            rollout_metrics = {
                "n_success": 3,
                "n_with_verifier_reward": 2,
                "has_verifier_reward": [1, 0, 1],
            }

        metrics = trainer.inject_lineage_loss_weight_into_config(config, FakeBatch())

        algorithm = config["algorithm"]
        assert isinstance(algorithm, dict)
        assert algorithm["lineage_loss_weight"] == pytest.approx(4.0 / 3.0)
        assert algorithm["lineage_loss_weights"] == [1.5, 1.0, 1.5]
        assert metrics == {
            "lineage_loss_weight": pytest.approx(4.0 / 3.0),
            "lineage_loss_weight_min": pytest.approx(1.0),
            "lineage_loss_weight_max": pytest.approx(1.5),
        }

    def test_inject_lineage_loss_weight_into_object_algorithm_config(self) -> None:
        trainer = _CoreTrainer()
        trainer.init_gfn()

        class Algorithm:
            lineage_loss_weight: float
            lineage_loss_weights: list[float] | None

            def __init__(self) -> None:
                self.lineage_weighting = {
                    "metric": "n_with_verifier_reward",
                    "denominator_metric": "n_success",
                    "base_weight": 1.0,
                    "scale": 1.0,
                }

        class Config:
            def __init__(self) -> None:
                self.algorithm = Algorithm()

        class FakeBatch:
            rollout_metrics = {
                "n_success": 2,
                "n_with_verifier_reward": 1,
                "has_verifier_reward": [1, 0],
            }

        config = Config()
        metrics = trainer.inject_lineage_loss_weight_into_config(config, FakeBatch())

        assert config.algorithm.lineage_loss_weight == pytest.approx(1.5)
        assert config.algorithm.lineage_loss_weights == [2.0, 1.0]
        assert metrics["lineage_loss_weight"] == pytest.approx(1.5)
        assert metrics["lineage_loss_weight_min"] == pytest.approx(1.0)
        assert metrics["lineage_loss_weight_max"] == pytest.approx(2.0)


class TestResolveLogzParams:
    def test_reads_dict_values(self) -> None:
        assert resolve_logz_params({"logZ_init": 2.0, "logZ_lr": 0.01}) == (2.0, 0.01)

    def test_reads_object_values(self) -> None:
        class Config:
            logZ_init = -1.0
            logZ_lr = 0.05

        assert resolve_logz_params(Config()) == (-1.0, 0.05)

    def test_uses_defaults_when_absent(self) -> None:
        init, lr = resolve_logz_params(object())
        assert init == pytest.approx(0.0)
        assert lr > 0.0
