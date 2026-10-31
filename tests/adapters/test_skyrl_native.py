import pytest
import torch
from omegaconf import OmegaConf

from synthstats.core.types import Message, Reward


class TestGFNLoss:
    def test_gfn_tb_loss_computes(self):
        from synthstats.integrations.skyrl.loss import gfn_tb_loss
        from synthstats.integrations.skyrl.trainer import LogZModule

        B, T = 2, 5
        log_probs = torch.randn(B, T)
        old_log_probs = torch.randn(B, T)
        # constant advantages per row (broadcasted log_reward)
        advantages = torch.tensor([[1.5] * T, [-0.3] * T])

        class FakeConfig:
            logZ_module = LogZModule(0.0)
            tb_max_residual = 100.0

        loss, metrics = gfn_tb_loss(
            log_probs,
            old_log_probs,
            advantages,
            FakeConfig(),
            loss_mask=torch.ones(B, T),
        )
        assert loss.shape == ()
        assert loss.requires_grad
        assert isinstance(metrics, dict)
        assert "clip_ratio" in metrics
        assert "tb_loss" in metrics
        assert "logZ" in metrics

    def test_gfn_tb_loss_skyrl_signature(self):
        from synthstats.integrations.skyrl.loss import gfn_tb_loss
        from synthstats.integrations.skyrl.trainer import LogZModule

        B, T = 3, 4
        log_probs = torch.randn(B, T)
        old_log_probs = torch.randn(B, T)
        advantages = torch.ones(B, T) * 2.0

        class FakeConfig:
            logZ_module = LogZModule(0.0)
            tb_max_residual = 100.0

        result = gfn_tb_loss(log_probs, old_log_probs, advantages, FakeConfig())
        assert isinstance(result, tuple)
        assert len(result) == 2
        loss, second = result
        assert isinstance(loss, torch.Tensor)
        assert isinstance(second, dict)
        assert "clip_ratio" in second

    def test_gfn_tb_loss_respects_mask(self):
        from synthstats.integrations.skyrl.loss import gfn_tb_loss
        from synthstats.integrations.skyrl.trainer import LogZModule

        B, T = 2, 5
        log_probs = torch.randn(B, T)
        old_log_probs = torch.randn(B, T)
        advantages = torch.ones(B, T)

        # mask out all tokens for trajectory 0
        loss_mask = torch.ones(B, T)
        loss_mask[0, :] = 0.0

        class FakeConfig:
            logZ_module = LogZModule(0.0)
            tb_max_residual = 100.0

        loss, _ = gfn_tb_loss(
            log_probs,
            old_log_probs,
            advantages,
            FakeConfig(),
            loss_mask=loss_mask,
        )
        assert torch.isfinite(loss)

    def test_gfn_tb_loss_applies_lineage_loss_weight(self):
        from synthstats.integrations.skyrl.loss import gfn_tb_loss
        from synthstats.integrations.skyrl.trainer import LogZModule

        log_probs = torch.tensor([[-1.0, -1.0]])
        advantages = torch.tensor([[2.0, 2.0]])
        loss_mask = torch.ones(1, 2)

        class BaseConfig:
            logZ_module = LogZModule(0.0)
            tb_max_residual = 100.0

        class WeightedConfig(BaseConfig):
            lineage_loss_weight = 1.5

        unweighted_loss, _ = gfn_tb_loss(
            log_probs, log_probs, advantages, BaseConfig(), loss_mask=loss_mask
        )
        weighted_loss, metrics = gfn_tb_loss(
            log_probs, log_probs, advantages, WeightedConfig(), loss_mask=loss_mask
        )

        assert torch.isclose(weighted_loss, unweighted_loss * 1.5)
        assert metrics["lineage_loss_weight"] == pytest.approx(1.5)
        assert metrics["tb_loss_unweighted"] == pytest.approx(unweighted_loss.item())

    def test_gfn_tb_loss_applies_per_sample_lineage_weights(self):
        from synthstats.integrations.skyrl.loss import gfn_tb_loss
        from synthstats.integrations.skyrl.trainer import LogZModule

        log_probs = torch.tensor(
            [
                [-1.0, -1.0],
                [-2.0, -2.0],
            ]
        )
        advantages = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )
        loss_mask = torch.ones(2, 2)

        class WeightedConfig:
            logZ_module = LogZModule(0.0)
            tb_max_residual = 100.0
            lineage_loss_weights = [2.0, 1.0]

        weighted_loss, metrics = gfn_tb_loss(
            log_probs, log_probs, advantages, WeightedConfig(), loss_mask=loss_mask
        )

        expected = torch.tensor((2.0 * 9.0 + 1.0 * 25.0) / 2.0)
        assert torch.isclose(weighted_loss, expected)
        assert metrics["lineage_loss_weight"] == pytest.approx(1.5)
        assert metrics["lineage_loss_weight_min"] == pytest.approx(1.0)
        assert metrics["lineage_loss_weight_max"] == pytest.approx(2.0)

    def test_gfn_tb_loss_filters_per_sample_weights_with_zero_mask_rows(self):
        from synthstats.integrations.skyrl.loss import gfn_tb_loss
        from synthstats.integrations.skyrl.trainer import LogZModule

        log_probs = torch.tensor(
            [
                [-5.0, -5.0],
                [-1.0, -1.0],
            ]
        )
        advantages = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )
        loss_mask = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 1.0],
            ]
        )

        class WeightedConfig:
            logZ_module = LogZModule(0.0)
            tb_max_residual = 100.0
            lineage_loss_weights = [10.0, 1.0]

        weighted_loss, metrics = gfn_tb_loss(
            log_probs, log_probs, advantages, WeightedConfig(), loss_mask=loss_mask
        )

        assert torch.isclose(weighted_loss, torch.tensor(9.0))
        assert metrics["lineage_loss_weight"] == pytest.approx(1.0)
        assert metrics["lineage_loss_weight_min"] == pytest.approx(1.0)
        assert metrics["lineage_loss_weight_max"] == pytest.approx(1.0)

    def test_gfn_tb_loss_zero_mask_rows(self):
        from synthstats.integrations.skyrl.loss import gfn_tb_loss
        from synthstats.integrations.skyrl.trainer import LogZModule

        B, T = 3, 5
        log_probs = torch.randn(B, T)
        old_log_probs = torch.randn(B, T)
        advantages = torch.ones(B, T) * 0.5

        loss_mask = torch.ones(B, T)
        loss_mask[0, :] = 0.0  # entirely zero row
        loss_mask[2, :] = 0.0  # another entirely zero row

        class FakeConfig:
            logZ_module = LogZModule(0.0)
            tb_max_residual = 100.0

        loss, _ = gfn_tb_loss(
            log_probs,
            old_log_probs,
            advantages,
            FakeConfig(),
            loss_mask=loss_mask,
        )
        assert torch.isfinite(loss)
        assert loss.requires_grad

    def test_gfn_tb_loss_errors_without_logz_module(self):
        from synthstats.integrations.skyrl.loss import gfn_tb_loss

        with pytest.raises(RuntimeError, match="logZ_module"):
            gfn_tb_loss(
                torch.randn(2, 5),
                torch.randn(2, 5),
                torch.randn(2, 5),
                object(),
            )

    def test_gfn_subtb_loss_rejects_missing_eos(self):
        from synthstats.integrations.skyrl.loss import gfn_subtb_loss
        from synthstats.integrations.skyrl.trainer import LogZModule

        B, T = 2, 5

        class FakeConfig:
            logZ_module = LogZModule(0.0)
            tb_max_residual = 100.0
            _eos_logprobs = None

        with pytest.raises(ValueError, match="requires aligned current-policy EOS"):
            gfn_subtb_loss(
                torch.randn(B, T),
                torch.randn(B, T),
                torch.ones(B, T),
                FakeConfig(),
                loss_mask=torch.ones(B, T),
            )

    def test_gfn_subtb_loss_with_eos(self):
        from synthstats.integrations.skyrl.loss import gfn_subtb_loss
        from synthstats.integrations.skyrl.trainer import LogZModule

        B, T = 2, 5

        class FakeConfig:
            logZ_module = LogZModule(0.0)
            tb_max_residual = 100.0
            _eos_logprobs = torch.randn(B, T)
            subtb_lambda = 0.9

        loss, clip_ratio = gfn_subtb_loss(
            torch.randn(B, T),
            torch.randn(B, T),
            torch.ones(B, T),
            FakeConfig(),
            loss_mask=torch.ones(B, T),
        )
        assert torch.isfinite(loss)
        assert isinstance(clip_ratio, dict)

    def test_gfn_subtb_loss_applies_per_sample_lineage_weights(self):
        from synthstats.integrations.skyrl.loss import gfn_subtb_loss
        from synthstats.integrations.skyrl.trainer import LogZModule
        from synthstats.train.objectives.losses import compute_modified_subtb_core_terms

        log_probs = torch.tensor(
            [
                [-0.2, -0.4],
                [-0.5, -0.1],
            ]
        )
        eos_logprobs = torch.tensor(
            [
                [-0.3, -0.2],
                [-0.2, -0.6],
            ]
        )
        advantages = torch.tensor(
            [
                [1.0, 1.0],
                [0.5, 0.5],
            ]
        )
        loss_mask = torch.ones(2, 2)

        class WeightedConfig:
            logZ_module = LogZModule(0.0)
            tb_max_residual = 100.0
            _eos_logprobs = eos_logprobs
            subtb_lambda = 0.9
            lineage_loss_weights = [2.0, 1.0]

        weighted_loss, metrics = gfn_subtb_loss(
            log_probs, log_probs, advantages, WeightedConfig(), loss_mask=loss_mask
        )
        numerators, denominators = compute_modified_subtb_core_terms(
            log_probs=log_probs,
            eos_logprobs=eos_logprobs,
            log_rewards=advantages,
            mask=loss_mask,
            subtb_lambda=0.9,
            max_residual=100.0,
        )
        expected = (numerators * torch.tensor([2.0, 1.0])).sum() / (
            denominators * torch.tensor([2.0, 1.0])
        ).sum().clamp(min=1.0)

        assert torch.isclose(weighted_loss, expected)
        assert metrics["lineage_loss_weight"] == pytest.approx(1.5)
        assert metrics["lineage_loss_weight_min"] == pytest.approx(1.0)
        assert metrics["lineage_loss_weight_max"] == pytest.approx(2.0)

    def test_gfn_subtb_loss_zero_mask_rows_with_eos(self):
        """eos_logprobs must be filtered alongside log_probs when rows are dropped."""
        from synthstats.integrations.skyrl.loss import gfn_subtb_loss
        from synthstats.integrations.skyrl.trainer import LogZModule

        B, T = 3, 5

        loss_mask = torch.ones(B, T)
        loss_mask[0, :] = 0.0  # entirely zero row

        class FakeConfig:
            logZ_module = LogZModule(0.0)
            tb_max_residual = 100.0
            _eos_logprobs = torch.randn(B, T)
            subtb_lambda = 0.9

        loss, clip_ratio = gfn_subtb_loss(
            torch.randn(B, T),
            torch.randn(B, T),
            torch.ones(B, T),
            FakeConfig(),
            loss_mask=loss_mask,
        )
        assert torch.isfinite(loss)
        assert isinstance(clip_ratio, dict)

    def test_register_gfn_losses_no_skyrl(self):
        from synthstats.integrations.skyrl.loss import register_gfn_losses

        result = register_gfn_losses()
        assert isinstance(result, bool)


class TestLogZModule:
    def test_forward(self):
        from synthstats.integrations.skyrl.trainer import LogZModule

        m = LogZModule(init_value=1.5)
        assert m().item() == pytest.approx(1.5)
        assert m.logZ.requires_grad

    def test_gradient_flows(self):
        from synthstats.integrations.skyrl.trainer import LogZModule

        m = LogZModule(0.0)
        loss = (m() - 5.0) ** 2
        loss.backward()
        assert m.logZ.grad is not None


def _core_trainer(config=None, *, logZ_init=0.0, logZ_lr=0.01):
    from synthstats.train.gfn_core import GFlowNetTrainerCore, resolve_logz_params

    trainer = GFlowNetTrainerCore()
    if config is not None:
        logZ_init, logZ_lr = resolve_logz_params(config)
    trainer.init_gfn(logZ_init, logZ_lr)
    return trainer


class TestGFlowNetTrainerCore:
    def test_standalone_init(self):
        trainer = _core_trainer(logZ_init=1.0, logZ_lr=0.01)
        assert trainer.logZ.item() == pytest.approx(1.0)

    def test_standalone_init_no_args(self):
        trainer = _core_trainer()
        assert trainer.logZ.item() == pytest.approx(0.0)
        assert isinstance(trainer.logZ, torch.Tensor)
        assert trainer.logZ.requires_grad

    def test_standalone_init_with_config_dict(self):
        cfg = {"logZ_init": 2.5, "logZ_lr": 0.001}
        trainer = _core_trainer(config=cfg)
        assert trainer.logZ.item() == pytest.approx(2.5)

    def test_standalone_init_with_config_object(self):
        class FakeConfig:
            logZ_init = -1.0
            logZ_lr = 0.05

        trainer = _core_trainer(config=FakeConfig())
        assert trainer.logZ.item() == pytest.approx(-1.0)

    def test_logz_is_nn_parameter(self):
        trainer = _core_trainer(logZ_init=1.0)
        assert isinstance(trainer.logZ_module.logZ, torch.nn.Parameter)
        assert trainer.logZ_module.logZ.requires_grad

    def test_logz_in_optimizer(self):
        trainer = _core_trainer(logZ_init=2.0, logZ_lr=0.05)
        assert len(list(trainer.logZ_module.parameters())) == 1

    def test_gfn_state_dict_roundtrip(self):
        trainer = _core_trainer(logZ_init=3.0)
        state = trainer.gfn_state_dict()
        assert state["logZ"] == pytest.approx(3.0)
        assert "logZ_optimizer" in state
        assert "gfn_step_count" in state
        assert state["gfn_step_count"] == 0

        trainer2 = _core_trainer(logZ_init=0.0)
        trainer2.gfn_load_state_dict(state)
        assert trainer2.logZ.item() == pytest.approx(3.0)
        assert trainer2._gfn_step_count == 0

    def test_gfn_state_dict_roundtrip_after_steps(self):
        trainer = _core_trainer(logZ_init=1.0, logZ_lr=0.1)
        # simulate gradient on logZ then step
        loss = (trainer.logZ - 5.0) ** 2
        loss.backward()
        metrics = trainer.gfn_optimizer_step()
        assert metrics["gfn_step"] == 1
        assert metrics["logZ"] != pytest.approx(1.0)  # logZ moved

        state = trainer.gfn_state_dict()
        assert state["gfn_step_count"] == 1

        trainer2 = _core_trainer(logZ_init=0.0)
        trainer2.gfn_load_state_dict(state)
        assert trainer2.logZ.item() == pytest.approx(state["logZ"])
        assert trainer2._gfn_step_count == 1

    def test_inject_logZ_into_dict_config(self):
        trainer = _core_trainer(logZ_init=5.0)
        config: dict = {}
        trainer.inject_logZ_into_config(config)
        assert config["logZ"] == pytest.approx(5.0)

    def test_inject_logZ_into_dict_config_with_algorithm(self):
        trainer = _core_trainer(logZ_init=7.0)
        config: dict = {"algorithm": {}}
        trainer.inject_logZ_into_config(config)
        assert config["algorithm"]["logZ"] == pytest.approx(7.0)

    def test_inject_logZ_into_object_config(self):
        class FakeConfig:
            logZ: float
            logZ_module: object

        trainer = _core_trainer(logZ_init=4.0)
        cfg = FakeConfig()
        trainer.inject_logZ_into_config(cfg)
        assert cfg.logZ == pytest.approx(4.0)
        assert hasattr(cfg, "logZ_module")

    def test_extract_numeric_rollout_metrics_from_rollout_metrics_attr(self):
        from synthstats.integrations.skyrl.trainer import extract_numeric_rollout_metrics

        class FakeBatch:
            rollout_metrics = {
                "n_success": 2,
                "n_with_trace_id": 1,
                "reward_sources": ["verifier"],
                "reward_evidence_kinds": ["verifier"],
                "reward_truth_owners": ["verifier"],
            }

        assert extract_numeric_rollout_metrics(FakeBatch()) == {
            "n_success": 2.0,
            "n_with_trace_id": 1.0,
            "n_reward_evidence_kind_verifier": 1.0,
            "reward_evidence_kind_verifier_fraction": 1.0,
            "n_reward_truth_owner_verifier": 1.0,
            "reward_truth_owner_verifier_fraction": 1.0,
        }

    def test_extract_numeric_rollout_metrics_from_non_tensor_batch(self):
        from synthstats.integrations.skyrl.trainer import extract_numeric_rollout_metrics

        class FakeBatch:
            non_tensor_batch = {
                "rollout_metrics": {
                    "n_failed": 3,
                    "n_with_verifier_reward": 2,
                    "reward_evidence_kinds": ["verifier", "heuristic"],
                    "reward_truth_owners": ["verifier", "heuristic"],
                }
            }

        assert extract_numeric_rollout_metrics(FakeBatch()) == {
            "n_failed": 3.0,
            "n_with_verifier_reward": 2.0,
            "n_reward_evidence_kind_heuristic": 1.0,
            "n_reward_evidence_kind_verifier": 1.0,
            "reward_evidence_kind_heuristic_fraction": 0.5,
            "reward_evidence_kind_verifier_fraction": 0.5,
            "n_reward_truth_owner_heuristic": 1.0,
            "n_reward_truth_owner_verifier": 1.0,
            "reward_truth_owner_heuristic_fraction": 0.5,
            "reward_truth_owner_verifier_fraction": 0.5,
        }

    def test_extract_rollout_metric_sequences_from_rollout_metrics_attr(self):
        from synthstats.integrations.skyrl.trainer import extract_rollout_metric_sequences

        class FakeBatch:
            rollout_metrics = {
                "has_verifier_reward": [1, 0, 1],
                "reward_sources": ["verifier", "judge", "verifier"],
            }

        assert extract_rollout_metric_sequences(FakeBatch()) == {
            "has_verifier_reward": [1.0, 0.0, 1.0],
        }

    def test_compute_lineage_loss_weights_uses_rollout_flags(self):
        from synthstats.integrations.skyrl.trainer import compute_lineage_loss_weights

        class FakeConfig:
            algorithm = {
                "lineage_weighting": {
                    "metric": "n_with_verifier_reward",
                    "denominator_metric": "n_success",
                    "base_weight": 1.0,
                    "scale": 0.5,
                    "min_weight": 1.0,
                    "max_weight": 2.0,
                }
            }

        class FakeBatch:
            rollout_metrics = {
                "n_success": 3,
                "n_with_verifier_reward": 2,
                "has_verifier_reward": [1, 0, 1],
            }

        assert compute_lineage_loss_weights(FakeBatch(), FakeConfig()) == [1.5, 1.0, 1.5]

    def test_compute_lineage_loss_weight_uses_metric_fraction(self):
        from synthstats.integrations.skyrl.trainer import compute_lineage_loss_weight

        class FakeConfig:
            algorithm = {
                "lineage_weighting": {
                    "metric": "n_with_verifier_reward",
                    "denominator_metric": "n_success",
                    "base_weight": 1.0,
                    "scale": 2.0,
                    "min_weight": 1.0,
                    "max_weight": 3.0,
                }
            }

        class FakeBatch:
            rollout_metrics = {
                "n_success": 4,
                "n_with_verifier_reward": 2,
            }

        assert compute_lineage_loss_weight(FakeBatch(), FakeConfig()) == pytest.approx(2.0)

    def test_compute_lineage_loss_weight_averages_per_sample_weights(self):
        from synthstats.integrations.skyrl.trainer import compute_lineage_loss_weight

        class FakeConfig:
            algorithm = {
                "lineage_weighting": {
                    "metric": "n_with_verifier_reward",
                    "denominator_metric": "n_success",
                    "base_weight": 1.0,
                    "scale": 0.5,
                }
            }

        class FakeBatch:
            rollout_metrics = {
                "n_success": 3,
                "n_with_verifier_reward": 2,
                "has_verifier_reward": [1, 0, 1],
            }

        assert compute_lineage_loss_weight(FakeBatch(), FakeConfig()) == pytest.approx(
            (1.5 + 1.0 + 1.5) / 3.0
        )

    def test_compute_lineage_batch_order_groups_positive_rollouts_first(self):
        from synthstats.integrations.skyrl.trainer import compute_lineage_batch_order

        class FakeConfig:
            algorithm = {
                "lineage_batching": {
                    "metric": "n_with_verifier_reward",
                    "positive_first": True,
                }
            }

        class FakeBatch:
            rollout_metrics = {
                "has_verifier_reward": [0, 1, 0, 1],
            }

        assert compute_lineage_batch_order(FakeBatch(), FakeConfig()) == [1, 3, 0, 2]

    def test_apply_lineage_batching_reorders_tensors_and_rollout_metadata(self):
        from synthstats.integrations.skyrl.trainer import apply_lineage_batching

        class FakeConfig:
            algorithm = {
                "lineage_batching": {
                    "metric": "n_with_verifier_reward",
                    "positive_first": True,
                }
            }

        class FakeBatch:
            response_ids = torch.tensor([[10], [20], [30], [40]])
            rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
            non_tensor_batch = {
                "rollout_metrics": {
                    "n_success": 4,
                    "n_with_verifier_reward": 2,
                    "has_verifier_reward": [0, 1, 0, 1],
                    "reward_sources": ["heuristic", "verifier", "heuristic", "verifier"],
                    "reward_evidence_kinds": ["heuristic", "verifier", "heuristic", "verifier"],
                    "reward_truth_owners": ["heuristic", "verifier", "heuristic", "verifier"],
                }
            }

        batch = FakeBatch()
        metrics = apply_lineage_batching(batch, FakeConfig())

        assert batch.response_ids.squeeze(-1).tolist() == [20, 40, 10, 30]
        assert batch.rewards.tolist() == [2.0, 4.0, 1.0, 3.0]
        assert batch.non_tensor_batch["rollout_metrics"]["has_verifier_reward"] == [1, 1, 0, 0]
        assert batch.non_tensor_batch["rollout_metrics"]["reward_sources"] == [
            "verifier",
            "verifier",
            "heuristic",
            "heuristic",
        ]
        assert batch.non_tensor_batch["rollout_metrics"]["reward_evidence_kinds"] == [
            "verifier",
            "verifier",
            "heuristic",
            "heuristic",
        ]
        assert batch.non_tensor_batch["rollout_metrics"]["reward_truth_owners"] == [
            "verifier",
            "verifier",
            "heuristic",
            "heuristic",
        ]
        assert metrics["lineage_batch_reordered"] == pytest.approx(1.0)
        assert metrics["lineage_batch_positive_count"] == pytest.approx(2.0)
        assert metrics["lineage_batch_negative_count"] == pytest.approx(2.0)
        assert metrics["lineage_batch_positive_first"] == pytest.approx(1.0)

    def test_validate_gfn_algorithm_config_rejects_non_gfn_loss(self):
        from synthstats.integrations.skyrl.trainer import validate_gfn_algorithm_config

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
        from synthstats.integrations.skyrl.trainer import validate_gfn_algorithm_config

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


class TestTools:
    def test_tool_definitions(self):
        from synthstats.train.agent_tools import SYNTHSTATS_TOOLS

        assert len(SYNTHSTATS_TOOLS) == 3
        names = [t["function"]["name"] for t in SYNTHSTATS_TOOLS]
        assert "query" in names
        assert "compute_eig" in names
        assert "submit_program" in names

    def test_tool_schema_structure(self):
        from synthstats.train.agent_tools import SYNTHSTATS_TOOLS

        for tool in SYNTHSTATS_TOOLS:
            assert "type" in tool
            assert "function" in tool
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func


class TestReward:
    def test_reward_function_calls_judge(self):
        from synthstats.train.reward import SynthStatsRewardFunction

        class MockJudge:
            def score(self, *, task_name, trajectory, artifacts):
                return Reward(total=42.0, components={}, info={})

        rf = SynthStatsRewardFunction(MockJudge(), task_name="test_task")
        result = rf([{"role": "user", "content": "hi"}])
        assert result == 42.0

    def test_reward_function_passes_task_name(self):
        from synthstats.train.reward import SynthStatsRewardFunction

        received = {}

        class MockJudge:
            def score(self, *, task_name, trajectory, artifacts):
                received["task_name"] = task_name
                return Reward(total=1.0, components={}, info={})

        rf = SynthStatsRewardFunction(MockJudge(), task_name="dugongs")
        rf([{"role": "user", "content": "hi"}])
        assert received["task_name"] == "dugongs"

    def test_score_with_details(self):
        from synthstats.train.reward import SynthStatsRewardFunction

        class MockJudge:
            def score(self, *, task_name, trajectory, artifacts):
                return Reward(
                    total=10.0,
                    components={"elpd": 10.0},
                    info={"samples": 100},
                )

        rf = SynthStatsRewardFunction(MockJudge(), task_name="test")
        reward = rf.score_with_details([{"role": "user", "content": "hi"}])
        assert reward.total == 10.0
        assert reward.components["elpd"] == 10.0


class TestAgentLoop:
    def test_agent_loop_output_dataclass(self):
        from synthstats.train.agent_loop import AgentLoopOutput

        out = AgentLoopOutput(
            prompt_ids=[1, 2],
            response_ids=[3, 4, 5],
            response_mask=[1, 1, 0],
            reward=1.5,
        )
        assert out.reward == 1.5
        assert len(out.response_mask) == 3

    def test_run_episode_basic(self):
        from synthstats.core.types import FinalAnswer, StepResult
        from synthstats.train.agent_loop import SynthStatsAgentLoop

        class MockTask:
            name = "test"

            def reset(self, seed=None):
                return {"step": 0}

            def observe(self, state):
                return [Message("user", "describe the data")]

            def step(self, state, action):
                return StepResult(
                    next_state={"step": 1},
                    done=True,
                    artifacts={"reward": 5.0},
                )

        class MockCodec:
            def parse(self, text):
                return FinalAnswer(text=text)

            def format(self, action):
                return str(action)

        def mock_generate(messages, temperature=0.7):
            return {
                "text": "my answer",
                "token_ids": [10, 20, 30],
                "prompt_ids": [1, 2],
            }

        loop = SynthStatsAgentLoop(
            task=MockTask(),
            codec=MockCodec(),
        )
        output = loop.run_episode(mock_generate, seed=42)

        assert output.reward == 5.0
        assert output.info["done"] is True
        assert output.info["turns"] == 1
        assert len(output.response_ids) == 3
        assert all(m == 1 for m in output.response_mask)

    def test_run_episode_with_judge_records_reward_details(self):
        from synthstats.core.types import FinalAnswer, StepResult
        from synthstats.train.agent_loop import SynthStatsAgentLoop

        class MockTask:
            name = "test"

            def reset(self, seed=None):
                return {"step": 0}

            def observe(self, state):
                return [Message("user", "describe the data")]

            def step(self, state, action):
                return StepResult(
                    next_state={"step": 1},
                    done=True,
                    artifacts={"program": "x = 1"},
                )

        class MockCodec:
            def parse(self, text):
                return FinalAnswer(text=text)

        class MockJudge:
            def score(self, *, task_name, trajectory, artifacts):
                assert trajectory.messages[-1].content == "my answer"
                return Reward(
                    total=2.0,
                    components={"elpd": 2.0},
                    info={"source": "judge"},
                )

        def mock_generate(messages, temperature=0.7):
            return {
                "text": "my answer",
                "token_ids": [10, 20, 30],
                "prompt_ids": [1, 2],
            }

        loop = SynthStatsAgentLoop(
            task=MockTask(),
            codec=MockCodec(),
            judge=MockJudge(),
        )
        output = loop.run_episode(mock_generate, seed=42)

        assert output.reward == pytest.approx(2.0)
        assert output.info["reward_components"] == {"elpd": 2.0}
        assert output.info["reward_info"] == {
            "source": "judge",
            "reward_source": "judge",
            "evidence_kind": "heuristic",
            "truth_owner": "heuristic",
            "raw_total": 2.0,
        }

    def test_run_episode_skips_native_tools_even_with_executor(self):
        from synthstats.core.types import FinalAnswer, StepResult, ToolCall
        from synthstats.train.agent_loop import SynthStatsAgentLoop

        class MockTask:
            name = "test"

            def reset(self, seed=None):
                return {"step": 0}

            def observe(self, state):
                return [Message("user", f"step-{state['step']}")]

            def step(self, state, action):
                done = isinstance(action, FinalAnswer)
                return StepResult(
                    next_state={"step": state["step"] + 1},
                    done=done,
                    artifacts={"reward": 5.0} if done else {},
                )

            def native_actions(self):
                return frozenset({"native_tool"})

        class MockCodec:
            def parse(self, text):
                if text == "native":
                    return ToolCall(name="native_tool", input={"x": 1}, raw=text)
                return FinalAnswer(text=text)

        class FailingExecutor:
            name = "native_tool"

            def execute(self, payload, *, timeout_s):
                raise AssertionError("native tools must not use executors")

        responses = iter(["native", "done"])

        def mock_generate(messages, temperature=0.7):
            text = next(responses)
            return {"text": text, "token_ids": [10], "prompt_ids": [1, 2]}

        loop = SynthStatsAgentLoop(
            task=MockTask(),
            codec=MockCodec(),
            executors={"native_tool": FailingExecutor()},
        )
        output = loop.run_episode(mock_generate, seed=42)

        assert output.info["tool_calls"] == 0
        assert [m for m in output.messages if m["role"] == "tool"] == []
        assert output.reward == 5.0

    def test_run_episode_surfaces_unknown_external_tool_errors(self):
        from synthstats.core.types import FinalAnswer, StepResult, ToolCall
        from synthstats.train.agent_loop import SynthStatsAgentLoop

        class MockTask:
            name = "test"

            def reset(self, seed=None):
                return {"step": 0}

            def observe(self, state):
                return [Message("user", f"step-{state['step']}")]

            def step(self, state, action):
                done = isinstance(action, FinalAnswer)
                return StepResult(
                    next_state={"step": state["step"] + 1},
                    done=done,
                    artifacts={"reward": 5.0} if done else {},
                )

            def native_actions(self):
                return frozenset({"native_tool"})

        class MockCodec:
            def parse(self, text):
                if text == "external":
                    return ToolCall(name="external_tool", input={"x": 1}, raw=text)
                return FinalAnswer(text=text)

        responses = iter(["external", "done"])

        def mock_generate(messages, temperature=0.7):
            text = next(responses)
            return {"text": text, "token_ids": [10], "prompt_ids": [1, 2]}

        loop = SynthStatsAgentLoop(task=MockTask(), codec=MockCodec(), executors={})
        output = loop.run_episode(mock_generate, seed=42)

        tool_messages = [m for m in output.messages if m["role"] == "tool"]
        assert tool_messages == [
            {
                "role": "tool",
                "content": "Error: Unknown tool 'external_tool'. Available tools: []",
            }
        ]
        assert output.info["tool_calls"] == 1
        assert output.info["tool_error"] == "Unknown tool: external_tool"
        assert output.reward == 5.0


class TestAdapters:
    def test_agent_loop_output_to_dict(self):
        from synthstats.train.agent_loop import AgentLoopOutput
        from synthstats.train.data.adapters import agent_loop_output_to_dict

        out = AgentLoopOutput(
            prompt_ids=[1, 2],
            response_ids=[3, 4, 5],
            response_mask=[1, 1, 0],
            reward=2.5,
        )
        d = agent_loop_output_to_dict(out)
        assert d["response_ids"].shape == (3,)
        assert d["response_mask"].shape == (3,)
        assert d["reward"].item() == pytest.approx(2.5)

    def test_batch_agent_loop_outputs(self):
        from synthstats.train.agent_loop import AgentLoopOutput
        from synthstats.train.data.adapters import batch_agent_loop_outputs

        outputs = [
            AgentLoopOutput(response_ids=[1, 2, 3], response_mask=[1, 1, 0], reward=1.0),
            AgentLoopOutput(response_ids=[4, 5], response_mask=[1, 1], reward=2.0),
        ]
        batch = batch_agent_loop_outputs(outputs, pad_id=0)
        assert batch["response_ids"].shape == (2, 3)  # padded to max len
        assert batch["response_mask"].shape == (2, 3)
        assert batch["rewards"].shape == (2,)
        assert batch["response_ids"][1, 2].item() == 0  # padded

    def test_batch_empty(self):
        from synthstats.train.data.adapters import batch_agent_loop_outputs

        batch = batch_agent_loop_outputs([])
        assert batch["response_ids"].shape == (0, 0)


class TestRunner:
    def test_skyrl_runner_requires_skyrl(self):
        from synthstats.integrations.skyrl.trainer import SKYRL_TRAINER_AVAILABLE

        if SKYRL_TRAINER_AVAILABLE:
            pytest.skip("SkyRL is installed, can't test missing import")

        from synthstats.integrations.skyrl.runner import SkyRLRunner

        with pytest.raises(ImportError, match="skyrl-train"):
            SkyRLRunner({})

    def test_skyrl_runner_missing_dependency_error_mentions_linux(self, monkeypatch):
        monkeypatch.setattr(
            "synthstats.integrations.skyrl.runner.platform.system", lambda: "Darwin"
        )
        from synthstats.integrations.skyrl.trainer import SKYRL_TRAINER_AVAILABLE

        if SKYRL_TRAINER_AVAILABLE:
            pytest.skip("SkyRL is installed, can't test missing import")

        from synthstats.integrations.skyrl.runner import SkyRLRunner

        with pytest.raises(ImportError) as excinfo:
            SkyRLRunner({})

        message = str(excinfo.value)
        assert "Linux" in message
        assert "uv sync --extra skyrl" in message
        assert "This repo" in message

    def _patch_runner_deps(self, monkeypatch, skyrl_runner, *, calls, trainer_cls):
        """Patch all SkyRL dependencies so the runner can instantiate and run."""

        class FakeBasePPOExp:
            def __init__(self, cfg):
                self.cfg = cfg
                self.trainer = None

            def run(self):
                self.trainer = trainer_cls(self.cfg)
                calls.append("exp.run")

        monkeypatch.setattr(skyrl_runner, "SKYRL_TRAINER_AVAILABLE", True)
        monkeypatch.setattr(skyrl_runner, "_BasePPOExp", FakeBasePPOExp)
        monkeypatch.setattr(skyrl_runner, "_initialize_ray", lambda cfg: calls.append("init_ray"))
        monkeypatch.setattr(skyrl_runner, "register_all", lambda: calls.append("register"))
        monkeypatch.setattr(skyrl_runner, "GFlowNetTrainer", trainer_cls)

    def _minimal_valid_cfg(self):
        return OmegaConf.create(
            {
                "runner": {
                    "type": "skyrl",
                    "data": {"train_data": ["train.parquet"]},
                    "trainer": {
                        "algorithm": {
                            "policy_loss_type": "gfn_tb",
                            "advantage_estimator": "tb_log_reward",
                        }
                    },
                }
            }
        )

    def test_skyrl_runner_uses_baseppoexp(self, monkeypatch):
        from synthstats.integrations.skyrl import runner as skyrl_runner

        calls: list[str] = []

        class FakeTrainer:
            def __init__(self, cfg):
                self.logZ = torch.tensor(1.25)

        self._patch_runner_deps(monkeypatch, skyrl_runner, calls=calls, trainer_cls=FakeTrainer)

        result = skyrl_runner.SkyRLRunner(self._minimal_valid_cfg()).run()

        assert "register" in calls
        assert "init_ray" in calls
        assert "exp.run" in calls
        # register_all must happen before initialize_ray
        assert calls.index("register") < calls.index("init_ray")
        assert result.error is None
        assert result.metrics["logZ"] == pytest.approx(1.25)

    def test_skyrl_runner_extracts_logz_from_trainer(self, monkeypatch):
        from synthstats.integrations.skyrl import runner as skyrl_runner

        calls: list[str] = []

        class FakeTrainer:
            def __init__(self, cfg):
                self.logZ = torch.tensor(3.75)

        self._patch_runner_deps(monkeypatch, skyrl_runner, calls=calls, trainer_cls=FakeTrainer)

        result = skyrl_runner.SkyRLRunner(self._minimal_valid_cfg()).run()

        assert result.error is None
        assert result.metrics["logZ"] == pytest.approx(3.75)

    def test_skyrl_runner_returns_numeric_trainer_metrics(self, monkeypatch):
        from synthstats.integrations.skyrl import runner as skyrl_runner

        calls: list[str] = []

        class FakeTrainer:
            def __init__(self, cfg):
                self.logZ = torch.tensor(2.5)
                self.all_metrics = {
                    "rollout_n_success": 4,
                    "rollout_n_with_trace_id": 3.0,
                    "reward_sources": ["verifier"],
                }

        self._patch_runner_deps(monkeypatch, skyrl_runner, calls=calls, trainer_cls=FakeTrainer)

        result = skyrl_runner.SkyRLRunner(self._minimal_valid_cfg()).run()

        assert result.error is None
        assert result.metrics["logZ"] == pytest.approx(2.5)
        assert result.metrics["rollout_n_success"] == pytest.approx(4.0)
        assert result.metrics["rollout_n_with_trace_id"] == pytest.approx(3.0)

    def test_skyrl_runner_captures_registry_failures(self, monkeypatch):
        from synthstats.integrations.skyrl import runner as skyrl_runner

        class FakeTrainer:
            def __init__(self, cfg):
                self.logZ = torch.tensor(0.0)

        class FakeBasePPOExp:
            pass

        monkeypatch.setattr(skyrl_runner, "SKYRL_TRAINER_AVAILABLE", True)
        monkeypatch.setattr(skyrl_runner, "_BasePPOExp", FakeBasePPOExp)
        monkeypatch.setattr(skyrl_runner, "_initialize_ray", lambda cfg: None)
        monkeypatch.setattr(
            skyrl_runner,
            "register_all",
            lambda: (_ for _ in ()).throw(RuntimeError("registry boom")),
        )
        monkeypatch.setattr(skyrl_runner, "GFlowNetTrainer", FakeTrainer)

        result = skyrl_runner.SkyRLRunner(self._minimal_valid_cfg()).run()

        assert result.error == "registry boom"

    def test_skyrl_runner_requires_baseppoexp(self, monkeypatch):
        from synthstats.integrations.skyrl import runner as skyrl_runner

        monkeypatch.setattr(skyrl_runner, "SKYRL_TRAINER_AVAILABLE", True)
        monkeypatch.setattr(skyrl_runner, "_BasePPOExp", None)
        monkeypatch.setattr(skyrl_runner, "_initialize_ray", lambda cfg: None)

        with pytest.raises(ImportError, match="BasePPOExp"):
            skyrl_runner.SkyRLRunner({})

    def test_skyrl_runner_requires_initialize_ray(self, monkeypatch):
        from synthstats.integrations.skyrl import runner as skyrl_runner

        class FakeBasePPOExp:
            pass

        monkeypatch.setattr(skyrl_runner, "SKYRL_TRAINER_AVAILABLE", True)
        monkeypatch.setattr(skyrl_runner, "_BasePPOExp", FakeBasePPOExp)
        monkeypatch.setattr(skyrl_runner, "_initialize_ray", None)

        with pytest.raises(ImportError, match="initialize_ray"):
            skyrl_runner.SkyRLRunner({})

    def test_skyrl_runner_typeerror_gives_constructor_diagnostic(self, monkeypatch):
        """exp.run() raising TypeError produces a diagnostic about constructor mismatch."""
        from synthstats.integrations.skyrl import runner as skyrl_runner

        class FakeBasePPOExp:
            def __init__(self, cfg):
                self.cfg = cfg

            def run(self):
                raise TypeError("__init__() got an unexpected keyword argument 'tracker'")

        monkeypatch.setattr(skyrl_runner, "SKYRL_TRAINER_AVAILABLE", True)
        monkeypatch.setattr(skyrl_runner, "_BasePPOExp", FakeBasePPOExp)
        monkeypatch.setattr(skyrl_runner, "_initialize_ray", lambda cfg: None)
        monkeypatch.setattr(skyrl_runner, "register_all", lambda: None)
        monkeypatch.setattr(skyrl_runner, "GFlowNetTrainer", type("T", (), {}))

        result = skyrl_runner.SkyRLRunner(self._minimal_valid_cfg()).run()

        assert result.error is not None
        assert "constructor mismatch" in result.error
        assert "tracker" in result.error
        assert "BasePPOExp._setup_trainer()" in result.error

    def test_skyrl_runner_works_when_skyrl_config_module_is_absent(self, monkeypatch):
        import sys

        from synthstats.integrations.skyrl import runner as skyrl_runner

        calls: list[str] = []

        class FakeTrainer:
            def __init__(self, cfg):
                self.logZ = torch.tensor(1.25)

        self._patch_runner_deps(monkeypatch, skyrl_runner, calls=calls, trainer_cls=FakeTrainer)
        monkeypatch.delitem(sys.modules, "synthstats.integrations.skyrl.config", raising=False)
        monkeypatch.setitem(sys.modules, "skyrl.train.config", None)
        monkeypatch.setitem(sys.modules, "skyrl.train.utils.utils", None)

        result = skyrl_runner.SkyRLRunner(self._minimal_valid_cfg()).run()
        assert result.error is None
        assert result.metrics["logZ"] == pytest.approx(1.25)

    def test_build_training_experiment_config_promotes_harbor_task_packages(self):
        from synthstats.train.experiment_config import build_training_experiment_config

        cfg = OmegaConf.create(
            {
                "backend": "vllm",
                "policy": {"model_name": "Qwen/Qwen3.5-4B"},
                "objective": {"loss_type": "subtb"},
                "env": {
                    "name": "dugongs",
                    "_target_": "synthstats.envs.task_runtime_env.TaskRuntimeEnv",
                    "task": {"env_name": "dugongs"},
                },
                "judge": {"_target_": "synthstats.judges.likelihood.LikelihoodJudge"},
                "runner": {
                    "type": "skyrl",
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
        assert exp_cfg.generator.inference_engine.backend == "vllm"
        assert exp_cfg.data.train_data == []

    def test_build_training_experiment_config_promotes_objective_lineage_weighting(self):
        from synthstats.train.experiment_config import build_training_experiment_config

        cfg = OmegaConf.create(
            {
                "backend": "vllm",
                "policy": {"model_name": "Qwen/Qwen3.5-4B"},
                "objective": {
                    "loss_type": "subtb",
                    "subtb_lambda": 0.85,
                    "tb_max_residual": 55.0,
                    "lineage_weighting": {
                        "metric": "n_with_verifier_reward",
                        "denominator_metric": "n_success",
                        "base_weight": 1.0,
                        "scale": 1.0,
                    },
                    "lineage_batching": {
                        "metric": "n_with_verifier_reward",
                        "positive_first": True,
                    },
                },
                "env": {
                    "name": "dugongs",
                    "_target_": "synthstats.envs.task_runtime_env.TaskRuntimeEnv",
                    "task": {"env_name": "dugongs"},
                },
                "judge": {"_target_": "synthstats.judges.likelihood.LikelihoodJudge"},
                "runner": {
                    "type": "skyrl",
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

        assert exp_cfg.trainer.algorithm.subtb_lambda == pytest.approx(0.85)
        assert exp_cfg.trainer.algorithm.tb_max_residual == pytest.approx(55.0)
        assert exp_cfg.trainer.algorithm.lineage_weighting.metric == "n_with_verifier_reward"
        assert exp_cfg.trainer.algorithm.lineage_batching.metric == "n_with_verifier_reward"

    def test_validate_training_experiment_config_requires_max_seq_len_for_harbor(self):
        from synthstats.train.experiment_config import validate_training_experiment_config

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
        from synthstats.train.experiment_config import validate_training_experiment_config

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


class TestMakePPOExpClass:
    def test_returns_none_without_skyrl(self, monkeypatch):
        from synthstats.integrations.skyrl import runner as skyrl_runner

        monkeypatch.setattr(skyrl_runner, "_BasePPOExp", None)
        result = skyrl_runner._make_ppo_exp_class(object)
        assert result is None

    def test_returns_class_with_baseppoexp(self, monkeypatch):
        from synthstats.integrations.skyrl import runner as skyrl_runner

        class FakeBasePPOExp:
            def __init__(self, cfg):
                self.cfg = cfg

        monkeypatch.setattr(skyrl_runner, "_BasePPOExp", FakeBasePPOExp)

        cls = skyrl_runner._make_ppo_exp_class(object)
        assert cls is not None
        assert issubclass(cls, FakeBasePPOExp)

    def test_get_trainer_delegates_to_trainer_cls(self, monkeypatch):
        from synthstats.integrations.skyrl import runner as skyrl_runner

        received_kwargs = {}

        class FakeBasePPOExp:
            def __init__(self, cfg):
                self.cfg = cfg

        class FakeTrainer:
            def __init__(self, **kwargs):
                received_kwargs.update(kwargs)

        monkeypatch.setattr(skyrl_runner, "_BasePPOExp", FakeBasePPOExp)

        cls = skyrl_runner._make_ppo_exp_class(FakeTrainer)
        instance = cls("my_cfg")

        # simulate what BasePPOExp._setup_trainer passes
        instance.get_trainer(
            cfg="cfg_val",
            tracker="tracker_val",
            tokenizer="tok_val",
        )
        assert received_kwargs["cfg"] == "cfg_val"
        assert received_kwargs["tracker"] == "tracker_val"
        assert received_kwargs["tokenizer"] == "tok_val"

    def test_get_generator_falls_back_to_super(self, monkeypatch):
        from synthstats.integrations.skyrl import runner as skyrl_runner

        class FakeBasePPOExp:
            def __init__(self, cfg):
                self.cfg = cfg

            def get_generator(self, cfg, tokenizer, inference_engine_client):
                assert (cfg, tokenizer, inference_engine_client) == ("cfg", "tok", "client")
                return "base_generator"

        monkeypatch.setattr(skyrl_runner, "_BasePPOExp", FakeBasePPOExp)

        cls = skyrl_runner._make_ppo_exp_class(object, generator_cls=None)
        instance = cls("cfg")
        assert instance.get_generator("cfg", "tok", "client") == "base_generator"

    def test_get_trainer_with_skyrl_kwargs(self, monkeypatch):
        """Verify the trainer class receives the full kwargs that SkyRL would pass."""
        from synthstats.integrations.skyrl import runner as skyrl_runner

        received = {}

        class FakeBasePPOExp:
            def __init__(self, cfg):
                self.cfg = cfg

        class FakeTrainer:
            def __init__(self, cfg, *args, **kwargs):
                received["cfg"] = cfg
                received["kwargs"] = kwargs

        monkeypatch.setattr(skyrl_runner, "_BasePPOExp", FakeBasePPOExp)

        cls = skyrl_runner._make_ppo_exp_class(FakeTrainer)
        instance = cls("my_cfg")

        # simulate the full set of kwargs from BasePPOExp._setup_trainer
        instance.get_trainer(
            cfg="the_config",
            tracker="the_tracker",
            tokenizer="the_tokenizer",
            train_dataset="train_ds",
            eval_dataset="eval_ds",
            inference_engine_client="iec",
            generator="gen",
            colocate_pg=None,
        )
        assert received["cfg"] == "the_config"
        assert received["kwargs"]["tracker"] == "the_tracker"
        assert received["kwargs"]["tokenizer"] == "the_tokenizer"
        assert received["kwargs"]["inference_engine_client"] == "iec"


class TestResolveLogZParams:
    def test_from_dict(self):
        from synthstats.train.gfn_core import resolve_logz_params

        init, lr = resolve_logz_params({"logZ_init": 2.0, "logZ_lr": 0.01})
        assert init == pytest.approx(2.0)
        assert lr == pytest.approx(0.01)

    def test_from_object(self):
        from synthstats.train.gfn_core import resolve_logz_params

        class Cfg:
            logZ_init = -3.0
            logZ_lr = 0.1

        init, lr = resolve_logz_params(Cfg())
        assert init == pytest.approx(-3.0)
        assert lr == pytest.approx(0.1)

    def test_defaults(self):
        from synthstats.core.constants import LOGZ_LR_DEFAULT
        from synthstats.train.gfn_core import resolve_logz_params

        init, lr = resolve_logz_params(object())
        assert init == pytest.approx(0.0)
        assert lr == pytest.approx(LOGZ_LR_DEFAULT)


class TestGFNNativeAdvantageAndMicrobatchReduction:
    def test_gfn_trainer_normalize_advantages_preserves_balanced_tb_targets(self):
        from synthstats.integrations.skyrl.loss import gfn_tb_loss
        from synthstats.integrations.skyrl.trainer import GFlowNetTrainer, LogZModule
        from synthstats.train.objectives.trajectory_balance import (
            compute_tb_log_reward_advantage,
        )

        # Four two-token trajectories: logPF = [-2, -4, -2, -4], logR = [-1, -3, -1, -3], logZ = 1.
        log_probs = torch.tensor(
            [
                [-1.0, -1.0],
                [-2.0, -2.0],
                [-1.0, -1.0],
                [-2.0, -2.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        response_mask = torch.ones((4, 2), dtype=torch.float32)
        loss_mask = torch.ones((4, 2), dtype=torch.float32)
        token_level_rewards = torch.tensor(
            [
                [0.0, float(torch.exp(torch.tensor(-1.0)).item())],
                [0.0, float(torch.exp(torch.tensor(-3.0)).item())],
                [0.0, float(torch.exp(torch.tensor(-1.0)).item())],
                [0.0, float(torch.exp(torch.tensor(-3.0)).item())],
            ],
            dtype=torch.float32,
        )

        advantages, returns = compute_tb_log_reward_advantage(
            token_level_rewards,
            response_mask,
            config={"reward_floor": 1e-8},
        )

        class SliceableBatch(dict):
            def __getitem__(self, key):
                if isinstance(key, slice):
                    return SliceableBatch({k: v[key] for k, v in self.items()})
                return super().__getitem__(key)

        data = SliceableBatch(
            {
                "advantages": advantages.clone(),
                "returns": returns.clone(),
                "response_mask": response_mask,
                "loss_mask": loss_mask,
            }
        )

        dummy_trainer = object.__new__(GFlowNetTrainer)
        dummy_trainer.cfg = OmegaConf.create(
            {
                "trainer": {
                    "micro_train_batch_size_per_gpu": 1,
                    "algorithm": {
                        "policy_loss_type": "gfn_tb",
                        "advantage_estimator": "tb_log_reward",
                        "advantage_batch_normalize": False,
                        "loss_reduction": "token_mean",
                        "max_seq_len": 8,
                    },
                }
            }
        )

        normalized = GFlowNetTrainer._normalize_advantages(
            dummy_trainer,
            data,
            mini_batch_boundaries=[(0, 4)],
            prompt_boundaries=[(0, 4)],
        )
        assert torch.allclose(normalized["advantages"], advantages, atol=1e-6)

        class LossCfg:
            logZ_module = LogZModule(1.0)
            tb_max_residual = 100.0

        loss, metrics = gfn_tb_loss(
            log_probs,
            log_probs.detach(),
            normalized["advantages"],
            LossCfg(),
            loss_mask=loss_mask,
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
        assert metrics["tb_loss"] == pytest.approx(0.0, abs=1e-6)
        loss.backward()
        assert log_probs.grad is not None
        assert torch.allclose(log_probs.grad, torch.zeros_like(log_probs), atol=1e-6)

    def test_gfn_tb_loss_microbatch_and_dp_partition_invariance(self):
        pytest.importorskip("skyrl")
        from types import SimpleNamespace

        from synthstats.integrations.skyrl.loss import gfn_tb_loss
        from synthstats.integrations.skyrl.policy_worker import GFNPolicyWorker
        from synthstats.integrations.skyrl.trainer import LogZModule

        # Three valid trajectories with residuals [-1, -3, -5] and one fully masked row:
        # squared residuals = [1, 9, 25], mean = 35/3, d/d(logZ) mean = (-2 - 6 - 10) / 3 = -6.
        base_log_probs = torch.tensor(
            [
                [-0.5, -0.5],  # sum = -1
                [-1.5, -1.5],  # sum = -3
                [-2.5, -2.5],  # sum = -5
                [-9.0, -9.0],  # masked row
            ],
            dtype=torch.float32,
        )
        base_advantages = torch.zeros((4, 2), dtype=torch.float32)
        base_mask = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        full_lp = base_log_probs.clone().requires_grad_(True)
        full_logz = LogZModule(0.0)

        class FullCfg:
            logZ_module = full_logz
            tb_max_residual = 100.0

        full_loss, full_metrics = gfn_tb_loss(
            full_lp, full_lp.detach(), base_advantages, FullCfg(), loss_mask=base_mask
        )
        full_loss.backward()
        assert full_loss.item() == pytest.approx(35.0 / 3.0, abs=1e-6)
        assert full_metrics["tb_loss"] == pytest.approx(35.0 / 3.0, abs=1e-6)
        assert full_logz.logZ.grad is not None
        assert full_logz.logZ.grad.item() == pytest.approx(-6.0, abs=1e-6)
        assert full_lp.grad is not None
        expected_lp_grad = full_lp.grad.clone()

        def _run_partitioned(
            rank_row_indices: list[list[int]], microbatch_size: int
        ) -> tuple[float, float, torch.Tensor]:
            from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
            from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase

            dp_size = len(rank_row_indices)
            local_valid_counts = [
                int((base_mask[rows].sum(dim=-1) > 0).sum().item()) for rows in rank_row_indices
            ]

            class _FakeStrategy:
                def all_reduce(self, metrics, op="sum", group=None):
                    assert op == "sum"
                    return {"valid_trajectories": float(sum(local_valid_counts))}

            class _FakeDeviceMesh:
                def get_group(self, name):
                    assert name == "dp"
                    return "dp_group"

            lp_param = base_log_probs.clone().requires_grad_(True)
            dp_losses: list[float] = []
            dp_logz_grads: list[torch.Tensor] = []

            for _rank, rows in enumerate(rank_row_indices):
                rank_logz = LogZModule(0.0)
                worker = GFNPolicyWorker.__new__(GFNPolicyWorker)
                worker.strategy = _FakeStrategy()
                worker.device_mesh = _FakeDeviceMesh()
                worker.mesh_rank = SimpleNamespace(dp_size=dp_size)
                worker.cfg = SimpleNamespace(
                    algorithm=SimpleNamespace(
                        logZ_module=rank_logz,
                        tb_max_residual=100.0,
                    )
                )

                def _fake_base_fb(self_worker, data, _rows=rows, **kwargs):
                    for start in range(0, len(_rows), microbatch_size):
                        mb_rows = _rows[start : start + microbatch_size]
                        mb_weight = len(mb_rows) / len(_rows)
                        self_worker._forward_backward_micro(
                            mb_rows,
                            mb_weight,
                            **kwargs,
                        )
                    return WorkerOutput()

                def _fake_base_fb_micro(self_worker, experience_rows, microbatch_weight, **kwargs):
                    mb_lp = lp_param[experience_rows]
                    policy_loss_r, metrics_r = gfn_tb_loss(
                        mb_lp,
                        mb_lp.detach(),
                        base_advantages[experience_rows],
                        self_worker.cfg.algorithm,
                        loss_mask=base_mask[experience_rows],
                    )
                    worker_loss = policy_loss_r * float(dp_size)
                    worker_loss.backward()
                    dp_losses.append(metrics_r["tb_loss"])
                    return metrics_r

                orig_fb = FSDPPolicyWorkerBase.forward_backward
                orig_fb_micro = FSDPPolicyWorkerBase._forward_backward_micro
                FSDPPolicyWorkerBase.forward_backward = _fake_base_fb
                FSDPPolicyWorkerBase._forward_backward_micro = _fake_base_fb_micro
                try:
                    worker.forward_backward({"loss_mask": base_mask[rows]})
                finally:
                    FSDPPolicyWorkerBase.forward_backward = orig_fb
                    FSDPPolicyWorkerBase._forward_backward_micro = orig_fb_micro

                assert rank_logz.logZ.grad is not None
                dp_logz_grads.append(rank_logz.logZ.grad.clone())

            avg_logz_grad = (sum(dp_logz_grads) / float(dp_size)).item()
            assert lp_param.grad is not None
            return sum(dp_losses), avg_logz_grad, (lp_param.grad / float(dp_size)).clone()

        # 1. Single rank, two microbatches with unequal valid counts ([0, 3] vs [1, 2])
        loss_mb, grad_mb, lp_grad_mb = _run_partitioned([[0, 3, 1, 2]], microbatch_size=2)
        assert loss_mb == pytest.approx(35.0 / 3.0, abs=1e-6)
        assert grad_mb == pytest.approx(-6.0, abs=1e-6)
        assert torch.allclose(lp_grad_mb, expected_lp_grad, atol=1e-6)

        # 2. Two DP ranks: Rank 1 has [-1, masked] (1 valid), Rank 2 has [-3, -5] (2 valid)
        loss_dp1, grad_dp1, lp_grad_dp1 = _run_partitioned([[0, 3], [1, 2]], microbatch_size=2)
        assert loss_dp1 == pytest.approx(35.0 / 3.0, abs=1e-6)
        assert grad_dp1 == pytest.approx(-6.0, abs=1e-6)
        assert torch.allclose(lp_grad_dp1, expected_lp_grad, atol=1e-6)

        # 3. Two DP ranks: Rank 1 has [-1, -3] (2 valid), Rank 2 has [-5, masked] (1 valid)
        loss_dp2, grad_dp2, lp_grad_dp2 = _run_partitioned([[0, 1], [2, 3]], microbatch_size=2)
        assert loss_dp2 == pytest.approx(35.0 / 3.0, abs=1e-6)
        assert grad_dp2 == pytest.approx(-6.0, abs=1e-6)
        assert torch.allclose(lp_grad_dp2, expected_lp_grad, atol=1e-6)

        # 4. Two DP ranks with an empty rank: Rank 1 has 3 valid, Rank 2 has all masked (0 valid)
        loss_empty, grad_empty, _ = _run_partitioned([[0, 1, 2], [3, 3, 3]], microbatch_size=2)
        assert loss_empty == pytest.approx(35.0 / 3.0, abs=1e-6)
        assert grad_empty == pytest.approx(-6.0, abs=1e-6)

    def test_validate_gfn_algorithm_config_rejects_advantage_batch_normalize(self):
        from synthstats.integrations.skyrl.trainer import validate_gfn_algorithm_config

        cfg = OmegaConf.create(
            {
                "trainer": {
                    "algorithm": {
                        "policy_loss_type": "gfn_tb",
                        "advantage_estimator": "tb_log_reward",
                        "advantage_batch_normalize": True,
                    }
                }
            }
        )
        with pytest.raises(ValueError, match="advantage_batch_normalize"):
            validate_gfn_algorithm_config(cfg)

    def test_gfn_policy_worker_propagates_and_restores_microbatch_reduction_context(
        self, monkeypatch
    ):
        pytest.importorskip("skyrl")
        from synthstats.integrations.skyrl.policy_worker import (
            FSDPPolicyWorkerBase,
            GFNPolicyWorker,
        )

        observed: list[tuple[float | None, int, int | None]] = []

        def fake_super_fb(
            self,
            data,
            loss_fn=None,
            loss_fn_config=None,
            return_per_token_outputs=True,
        ):
            from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput

            return WorkerOutput(
                metrics=self._forward_backward_micro(
                    experience=data,
                    microbatch_weight=0.25,
                    loss_fn=loss_fn,
                    loss_fn_config=loss_fn_config,
                    return_per_token_outputs=return_per_token_outputs,
                )
            )

        def fake_super_fb_micro(
            self,
            experience,
            microbatch_weight,
            loss_fn=None,
            loss_fn_config=None,
            return_per_token_outputs=True,
        ):
            alg = self.cfg.algorithm
            observed.append(
                (
                    alg.microbatch_weight,
                    alg.dp_size,
                    alg.minibatch_valid_trajectories,
                )
            )
            return {"policy_loss": 0.0}

        monkeypatch.setattr(FSDPPolicyWorkerBase, "forward_backward", fake_super_fb)
        monkeypatch.setattr(
            FSDPPolicyWorkerBase,
            "_forward_backward_micro",
            fake_super_fb_micro,
        )

        worker = object.__new__(GFNPolicyWorker)
        worker.cfg = type(
            "Cfg",
            (),
            {
                "algorithm": type(
                    "Alg",
                    (),
                    {
                        "microbatch_weight": None,
                        "dp_size": 1,
                        "minibatch_valid_trajectories": None,
                    },
                )()
            },
        )()
        worker.mesh_rank = type("MeshRank", (), {"dp_size": 2})()

        batch = {
            "loss_mask": torch.tensor(
                [[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
                dtype=torch.float32,
            )
        }
        with pytest.raises(RuntimeError, match="Global valid-trajectory count"):
            worker.forward_backward(batch)

        worker.strategy = type(
            "Strategy",
            (),
            {
                "all_reduce": staticmethod(
                    lambda metrics, op="sum", group=None: {
                        "valid_trajectories": metrics["valid_trajectories"] + 2.0
                    }
                )
            },
        )()
        worker.device_mesh = type("Mesh", (), {"get_group": staticmethod(lambda name: "dp")})()
        worker.forward_backward(batch)
        assert observed == [(0.25, 2, 5)]
        assert worker.cfg.algorithm.microbatch_weight is None
        assert worker.cfg.algorithm.dp_size == 1
        assert worker.cfg.algorithm.minibatch_valid_trajectories is None


class TestNativeConfigSemanticValidation:
    def test_skyrl_harbor_default_config_passes_native_validate_cfg(self):
        pytest.importorskip("skyrl")
        from pathlib import Path

        from skyrl.train.utils.utils import validate_cfg

        from synthstats.integrations.skyrl.config import build_native_config
        from synthstats.train.experiment_config import build_training_experiment_config

        configs_dir = Path(__file__).resolve().parents[2] / "src" / "synthstats" / "configs"
        runner_yaml = OmegaConf.load(configs_dir / "runner" / "skyrl_harbor.yaml")
        cfg = OmegaConf.create(
            {
                "backend": "vllm",
                "seed": 42,
                "run": {"name": "c2_check", "output_dir": "/tmp/c2_out"},
                "objective": {
                    "loss_type": "tb",
                    "logZ_init": 0.0,
                    "logZ_lr": 0.01,
                    "reward_floor": 1e-8,
                },
                "policy": {"model_name": "Qwen/Qwen3.5-0.8B"},
                "runner": runner_yaml,
            }
        )
        cfg.runner.data.train_data = ["train.parquet"]

        prepared = build_training_experiment_config(cfg)
        native_cfg = build_native_config(prepared)

        assert native_cfg.trainer.policy.language_model_only is True
        assert native_cfg.generator.inference_engine.language_model_only is True
        validate_cfg(native_cfg)

    def test_build_native_config_rejects_inconsistent_language_model_only(self):
        pytest.importorskip("skyrl")
        from synthstats.integrations.skyrl.config import build_native_config

        bad_cfg = OmegaConf.create(
            {
                "trainer": {
                    "strategy": "fsdp",
                    "algorithm": {
                        "policy_loss_type": "gfn_tb",
                        "advantage_estimator": "tb_log_reward",
                        "advantage_batch_normalize": False,
                        "max_seq_len": 4096,
                    },
                    "policy": {
                        "language_model_only": True,
                        "model": {"path": "Qwen/Qwen3.5-0.8B"},
                    },
                },
                "data": {"train_data": ["train.parquet"], "val_data": []},
                "generator": {
                    "type": "harbor",
                    "inference_engine": {
                        "backend": "vllm",
                        "language_model_only": False,
                    },
                },
            }
        )
        with pytest.raises(
            (AssertionError, ValueError),
            match="language_model_only should be set consistently",
        ):
            build_native_config(bad_cfg)


class TestTaskJudgeSeedAndLineageConsumers:
    def test_env_judge_and_seed_reach_normalized_config_and_generator_requests(self):
        from synthstats.integrations.harbor.generator import normalize_task_request
        from synthstats.integrations.harbor.task_registry import build_local_task_request
        from synthstats.train.experiment_config import build_training_experiment_config

        cfg_dugongs = OmegaConf.create(
            {
                "backend": "vllm",
                "seed": 101,
                "env": {"name": "dugongs", "task": {"env_name": "dugongs"}},
                "judge": {"_target_": "synthstats.judges.likelihood.LikelihoodJudge", "beta": 0.5},
                "runner": {
                    "type": "skyrl",
                    "generator": {"type": "harbor", "inference_engine": {}},
                },
            }
        )
        cfg_surgical = OmegaConf.create(
            {
                "backend": "vllm",
                "seed": 202,
                "env": {
                    "name": "surgical",
                    "task": {"env_name": "surgical"},
                    "config": {"evaluation_seed": 42},
                },
                "judge": {"_target_": "synthstats.judges.likelihood.LikelihoodJudge", "beta": 1.25},
                "runner": {
                    "type": "skyrl",
                    "generator": {"type": "harbor", "inference_engine": {}},
                },
            }
        )

        exp_dugongs = build_training_experiment_config(cfg_dugongs)
        exp_surgical = build_training_experiment_config(cfg_surgical)

        assert exp_dugongs.generator.task_config_overrides != (
            exp_surgical.generator.task_config_overrides
        )

        local_req = build_local_task_request(cfg_surgical)
        assert local_req["task_config"]["seed"] == 202
        assert local_req["task_config"]["env"]["config"]["evaluation_seed"] == 202

        dataset_row_request = {
            "task_name": "dugongs",
            "task_path": "/tmp/dugongs",
            "task_config": {
                "seed": 7,
                "env": {
                    "name": "dugongs",
                    "task": {"env_name": "dugongs"},
                    "config": {"evaluation_seed": 7},
                },
                "judge": {"beta": 0.1},
            },
        }
        merged = normalize_task_request(
            dataset_row_request,
            overrides=OmegaConf.to_container(
                exp_surgical.generator.task_config_overrides, resolve=True
            ),
        )
        assert merged.task_name == "surgical"
        assert merged.task_config["seed"] == 202
        assert merged.task_config["env"]["name"] == "surgical"
        assert merged.task_config["env"]["config"]["evaluation_seed"] == 202
        assert merged.task_config["judge"]["beta"] == pytest.approx(1.25)

    @pytest.mark.parametrize(
        ("judge_preset", "expected_cls_name"),
        [
            ("likelihood", "LikelihoodJudge"),
            ("hybrid_lm", "CompositeJudge"),
            ("llm_critic_glm5", "LLMCriticJudge"),
            ("composite", "CompositeJudge"),
        ],
    )
    def test_hydrated_judge_presets_instantiate_via_build_env(
        self,
        judge_preset: str,
        expected_cls_name: str,
    ):
        from hydra import compose, initialize_config_module

        from synthstats.cli.structured_config import register_train_schemas
        from synthstats.envs.builders import build_env
        from synthstats.integrations.harbor.generator import normalize_task_request
        from synthstats.integrations.harbor.task_registry import build_local_task_request
        from synthstats.train.experiment_config import build_training_experiment_config

        register_train_schemas()
        with initialize_config_module(version_base=None, config_module="synthstats.configs"):
            cfg = compose(config_name="train", overrides=[f"judge={judge_preset}"])

        exp_cfg = build_training_experiment_config(cfg)
        local_req = build_local_task_request(cfg)
        merged = normalize_task_request(
            local_req,
            overrides=OmegaConf.to_container(exp_cfg.generator.task_config_overrides, resolve=True),
        )
        env = build_env(OmegaConf.create(merged.task_config))
        assert env.judge is not None
        assert type(env.judge).__name__ == expected_cls_name
        if judge_preset == "composite":
            assert [type(j).__name__ for j, _ in env.judge.judges] == [
                "LikelihoodJudge",
                "FormattingJudge",
            ]
        elif judge_preset == "hybrid_lm":
            assert [type(j).__name__ for j, _ in env.judge.judges] == [
                "LikelihoodJudge",
                "LLMCriticJudge",
            ]
        elif judge_preset == "likelihood":
            assert env.judge.beta == pytest.approx(1.0)
            assert env.judge.loo_gate_mode == "strict"
        elif judge_preset == "llm_critic_glm5":
            assert env.judge.model_name == "openrouter/z-ai/glm-5"

    def test_gfn_trainer_applies_lineage_controls_at_batch_boundary(self):
        from synthstats.integrations.skyrl.trainer import GFlowNetTrainer

        class SliceableBatch(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.metadata = {
                    "rollout_metrics": {
                        "n_success": 4,
                        "n_with_verifier_reward": 2,
                        "has_verifier_reward": [0, 1, 0, 1],
                        "reward_sources": ["heuristic", "verifier", "heuristic", "verifier"],
                        "reward_evidence_kinds": ["heuristic", "verifier", "heuristic", "verifier"],
                        "reward_truth_owners": ["heuristic", "verifier", "heuristic", "verifier"],
                    }
                }

            def __getitem__(self, key):
                if isinstance(key, slice):
                    return SliceableBatch({k: v[key] for k, v in self.items()})
                return super().__getitem__(key)

        batch = SliceableBatch(
            {
                "advantages": torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32),
                "returns": torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32),
                "response_mask": torch.ones((4, 1), dtype=torch.float32),
                "loss_mask": torch.ones((4, 1), dtype=torch.float32),
            }
        )

        dummy_trainer = object.__new__(GFlowNetTrainer)
        dummy_trainer.all_metrics = {}
        dummy_trainer.cfg = OmegaConf.create(
            {
                "trainer": {
                    "micro_train_batch_size_per_gpu": 1,
                    "algorithm": {
                        "policy_loss_type": "gfn_tb",
                        "advantage_estimator": "tb_log_reward",
                        "advantage_batch_normalize": False,
                        "lineage_batching": {
                            "metric": "n_with_verifier_reward",
                            "positive_first": True,
                        },
                        "lineage_weighting": {
                            "metric": "n_with_verifier_reward",
                            "base_weight": 1.0,
                            "scale": 0.5,
                        },
                    },
                }
            }
        )

        out = GFlowNetTrainer._normalize_advantages(
            dummy_trainer,
            batch,
            mini_batch_boundaries=[(0, 4)],
            prompt_boundaries=None,
        )

        assert out["advantages"].squeeze(-1).tolist() == [2.0, 4.0, 1.0, 3.0]
        assert out.metadata["rollout_metrics"]["has_verifier_reward"] == [1, 1, 0, 0]
        assert list(dummy_trainer.cfg.trainer.algorithm.lineage_loss_weights) == [
            1.5,
            1.5,
            1.0,
            1.0,
        ]
        assert dummy_trainer.all_metrics["lineage_batch_reordered"] == pytest.approx(1.0)
        assert dummy_trainer.all_metrics["lineage_loss_weight"] == pytest.approx(1.25)

    def test_native_generate_to_worker_transports_and_consumes_lineage_controls(self):
        import asyncio

        pytest.importorskip("skyrl")
        from synthstats.integrations.skyrl.policy_worker import GFNPolicyWorker
        from synthstats.integrations.skyrl.trainer import GFlowNetTrainer
        from synthstats.train.gfn_core import LogZModule
        from synthstats.train.objectives.gfn_losses import gfn_tb_loss

        trainer = object.__new__(GFlowNetTrainer)
        trainer.all_metrics = {}
        trainer.tokenizer = type("Tok", (), {"pad_token_id": 0})()
        trainer.dispatch = type("Disp", (), {"get_lcm_dp_size": staticmethod(lambda: 2)})()
        trainer.cfg = OmegaConf.create(
            {
                "generator": {
                    "step_wise_trajectories": False,
                    "n_samples_per_prompt": 1,
                    "inference_engine": {"enable_return_routed_experts": False},
                },
                "trainer": {
                    "train_batch_size": 4,
                    "policy_mini_batch_size": 4,
                    "critic_mini_batch_size": 4,
                    "micro_train_batch_size_per_gpu": 1,
                    "critic": {"model": {"path": None}},
                    "algorithm": {
                        "policy_loss_type": "gfn_tb",
                        "advantage_estimator": "tb_log_reward",
                        "advantage_batch_normalize": False,
                        "loss_reduction": "token_mean",
                        "max_seq_len": 16,
                        "off_policy_correction": {
                            "tis_ratio_type": None,
                            "sequence_mask_metric": None,
                        },
                        "lineage_batching": {
                            "metric": "n_with_verifier_reward",
                            "positive_first": True,
                        },
                        "lineage_weighting": {
                            "metric": "n_with_verifier_reward",
                            "base_weight": 1.0,
                            "scale": 0.5,
                        },
                    },
                },
            }
        )

        class _FakeGenerator:
            async def generate(self, input_batch):
                del input_batch
                return {
                    "prompt_token_ids": [[1, 2], [1, 2], [1, 2], [1, 2]],
                    "response_ids": [[3], [4], [5], [6]],
                    "rewards": [[-1.0], [-3.0], [-5.0], [-7.0]],
                    "loss_masks": [[1], [1], [1], [1]],
                    "rollout_logprobs": None,
                    "stop_reasons": ["stop", "stop", "stop", "stop"],
                    "rollout_metrics": {
                        "n_success": 4,
                        "n_with_verifier_reward": 2,
                        "has_verifier_reward": [0, 1, 0, 1],
                        "reward_sources": ["heuristic", "verifier", "heuristic", "verifier"],
                        "reward_evidence_kinds": ["heuristic", "verifier", "heuristic", "verifier"],
                        "reward_truth_owners": ["heuristic", "verifier", "heuristic", "verifier"],
                    },
                }

        trainer.generator = _FakeGenerator()
        gen_out = asyncio.run(trainer.generate({"prompts": ["p0", "p1", "p2", "p3"]}))
        ti = trainer.convert_to_training_input(gen_out, uids=["u0", "u1", "u2", "u3"])
        assert "rollout_metrics" in ti.metadata

        ti["advantages"] = torch.tensor([[-1.0], [-3.0], [-5.0], [-7.0]], dtype=torch.float32)
        ti["action_log_probs"] = torch.zeros((4, 1), dtype=torch.float32)
        ti = trainer._normalize_advantages(ti, mini_batch_boundaries=[(0, 4)])

        # Verifier trajectories (-3.0, -7.0) must be ordered before heuristic (-1.0, -5.0).
        assert ti["advantages"].squeeze(-1).tolist() == pytest.approx([-3.0, -7.0, -1.0, -5.0])
        assert ti.metadata["policy_mini_batch_boundaries"] == [(0, 4)]
        assert ti.metadata["policy_prompt_boundaries"] == [(0, 1), (1, 2), (2, 3), (3, 4)]
        assert ti.metadata["lineage_loss_weights"] == pytest.approx([1.5, 1.5, 1.0, 1.0])
        assert "lineage_loss_weights" in ti
        assert ti["lineage_loss_weights"].tolist() == pytest.approx([1.5, 1.5, 1.0, 1.0])

        # Stage chunks across 2 DP ranks (2 trajectories per rank, 1 per microbatch).
        dp_chunks = ti.chunk(2)
        observed_mb_weights: list[list[float]] = []
        microbatch_losses: list[float] = []

        for dp_rank, dp_chunk in enumerate(dp_chunks):
            worker = object.__new__(GFNPolicyWorker)
            worker.mesh_rank = type("MeshRank", (), {"dp_size": 2, "dp": dp_rank})()
            worker.device_mesh = type("Mesh", (), {"get_group": staticmethod(lambda name: "dp")})()
            worker.strategy = type(
                "Strategy",
                (),
                {
                    "all_reduce": staticmethod(
                        lambda metrics, op="sum", group=None: (
                            {"valid_trajectories": 4.0}
                            if "valid_trajectories" in metrics
                            else dict(metrics)
                        )
                    )
                },
            )()
            worker.cfg = type(
                "WorkerCfg",
                (),
                {
                    "micro_train_batch_size_per_gpu": 1,
                    "max_tokens_per_microbatch": 0,
                    "algorithm": type(
                        "Alg",
                        (),
                        {
                            "policy_loss_type": "gfn_tb",
                            "logZ_module": LogZModule(0.0),
                            "logZ_lr": 1e-2,
                            "tb_max_residual": 100.0,
                            "microbatch_weight": None,
                            "dp_size": 1,
                            "minibatch_valid_trajectories": None,
                            "lineage_loss_weight": 1.0,
                            "lineage_loss_weights": None,
                        },
                    )(),
                },
            )()

            def _run_micro(
                self,
                experience,
                microbatch_weight,
                loss_fn=None,
                loss_fn_config=None,
                return_per_token_outputs=True,
            ):
                del microbatch_weight, loss_fn, loss_fn_config, return_per_token_outputs
                alg = self.cfg.algorithm
                lw = alg.lineage_loss_weights
                if isinstance(lw, torch.Tensor):
                    observed_mb_weights.append(lw.detach().cpu().tolist())
                elif isinstance(lw, list):
                    observed_mb_weights.append(list(lw))
                else:
                    observed_mb_weights.append([])
                loss, metrics = gfn_tb_loss(
                    log_probs=experience.action_log_probs,
                    old_log_probs=experience.action_log_probs,
                    advantages=experience.advantages,
                    config=alg,
                    loss_mask=experience.loss_mask,
                )
                microbatch_losses.append(float(loss.item()))
                return metrics

            from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase

            orig_micro = FSDPPolicyWorkerBase._forward_backward_micro
            FSDPPolicyWorkerBase._forward_backward_micro = _run_micro
            try:
                worker.forward_backward(dp_chunk)
            finally:
                FSDPPolicyWorkerBase._forward_backward_micro = orig_micro

            assert worker.cfg.algorithm.lineage_loss_weights is None
            assert worker.cfg.algorithm.lineage_loss_weight == pytest.approx(1.0)

            import inspect

            from skyrl.backends.skyrl_train.workers.worker_utils import (
                BaseBatchIterator,
                BatchIterator,
            )

            post_train_exp = next(BatchIterator(dp_chunk, sample_batch_size=1))
            assert post_train_exp.sequences.shape[0] == 1
            assert not hasattr(post_train_exp, "lineage_loss_weights")
            assert isinstance(
                inspect.getattr_static(BaseBatchIterator, "batch_to_experience"),
                staticmethod,
            )

        assert observed_mb_weights == [[1.5], [1.5], [1.0], [1.0]]
        # Exact weighted average: (1.5*9 + 1.5*49 + 1.0*1 + 1.0*25) / 4 = 113 / 4 = 28.25
        assert sum(microbatch_losses) == pytest.approx(28.25, abs=1e-5)

    def test_native_dynamic_sampling_replace_and_filter_preserve_lineage_alignment(self):
        import asyncio

        pytest.importorskip("skyrl")
        from synthstats.integrations.skyrl.trainer import GFlowNetTrainer

        def _make_trainer(sampling_type: str, train_batch_size: int = 2):
            trainer = object.__new__(GFlowNetTrainer)
            trainer.all_metrics = {}
            trainer.dynamic_sampling_state = None
            trainer.tokenizer = type("Tok", (), {"pad_token_id": 0})()
            trainer.dispatch = type("Disp", (), {"get_lcm_dp_size": staticmethod(lambda: 2)})()
            trainer.cfg = OmegaConf.create(
                {
                    "generator": {
                        "step_wise_trajectories": False,
                        "n_samples_per_prompt": 2,
                        "inference_engine": {"enable_return_routed_experts": False},
                    },
                    "trainer": {
                        "train_batch_size": train_batch_size,
                        "policy_mini_batch_size": train_batch_size,
                        "critic_mini_batch_size": train_batch_size,
                        "micro_train_batch_size_per_gpu": 1,
                        "critic": {"model": {"path": None}},
                        "algorithm": {
                            "policy_loss_type": "gfn_tb",
                            "advantage_estimator": "tb_log_reward",
                            "advantage_batch_normalize": False,
                            "loss_reduction": "token_mean",
                            "max_seq_len": 16,
                            "dynamic_sampling": {
                                "type": sampling_type,
                                "max_sample_batches": 4,
                                "min_replace_ratio": 0.0,
                            },
                            "off_policy_correction": {
                                "tis_ratio_type": None,
                                "sequence_mask_metric": None,
                            },
                            "lineage_weighting": {
                                "metric": "n_with_verifier_reward",
                                "base_weight": 1.0,
                                "scale": 0.5,
                            },
                        },
                    },
                }
            )
            return trainer

        # 1. Replace sampling: prompt A has non-zero reward spread (0, 1) and verifier flag 0;
        # prompt B has zero reward spread (0, 0) and verifier flag 1.
        # Replacement replaces B0, B1 with copies of A0, A1 -> all 4 rows come from A (flag 0).
        replace_trainer = _make_trainer("replace", train_batch_size=2)

        class _ReplaceGenerator:
            async def generate(self, input_batch):
                del input_batch
                return {
                    "prompt_token_ids": [[1, 2], [1, 2], [1, 3], [1, 3]],
                    "response_ids": [[3], [4], [5], [6]],
                    "rewards": [[0.0], [1.0], [0.0], [0.0]],
                    "loss_masks": [[1], [1], [1], [1]],
                    "rollout_logprobs": None,
                    "stop_reasons": ["stop", "stop", "stop", "stop"],
                    "rollout_metrics": {
                        "n_success": 4,
                        "n_with_verifier_reward": 2,
                        "has_verifier_reward": [0, 0, 1, 1],
                        "reward_sources": ["heuristic", "heuristic", "verifier", "verifier"],
                        "reward_evidence_kinds": ["heuristic", "heuristic", "verifier", "verifier"],
                        "reward_truth_owners": ["heuristic", "heuristic", "verifier", "verifier"],
                    },
                }

        replace_trainer.generator = _ReplaceGenerator()
        raw_out = asyncio.run(replace_trainer.generate({"prompts": ["A0", "A1", "B0", "B1"]}))
        replaced_out, replaced_uids, keep_sampling = replace_trainer.handle_dynamic_sampling(
            raw_out,
            ["A", "A", "B", "B"],
        )
        assert keep_sampling is False
        assert replaced_uids == ["A", "A", "A", "A"]

        ti_replace = replace_trainer.convert_to_training_input(replaced_out, replaced_uids)
        ti_replace["advantages"] = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float32)
        ti_replace["action_log_probs"] = torch.zeros((4, 1), dtype=torch.float32)
        ti_replace = replace_trainer._normalize_advantages(ti_replace)

        assert ti_replace.metadata["rollout_metrics"]["has_verifier_reward"] == [0, 0, 0, 0]
        assert ti_replace.metadata["rollout_metrics"]["n_with_verifier_reward"] == 0
        assert ti_replace["lineage_loss_weights"].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])

        # 2. Filter sampling across two generations with concatenation and truncation:
        # Generation 1: A (kept, verifier=[1, 1]), B (filtered out, verifier=[0, 0]) -> 1 prompt < 2
        # Generation 2: C (kept, verifier=[0, 0]), D (kept, verifier=[1, 1]) -> 3 prompts >= 2,
        # truncated to max_trajectories=4 -> [A0, A1, C0, C1] with verifier=[1, 1, 0, 0].
        filter_trainer = _make_trainer("filter", train_batch_size=2)
        batches = [
            {
                "prompt_token_ids": [[1, 2], [1, 2], [1, 3], [1, 3]],
                "response_ids": [[3], [4], [5], [6]],
                "rewards": [[0.0], [1.0], [0.0], [0.0]],
                "loss_masks": [[1], [1], [1], [1]],
                "rollout_logprobs": None,
                "stop_reasons": ["stop", "stop", "stop", "stop"],
                "rollout_metrics": {
                    "n_success": 4,
                    "n_with_verifier_reward": 2,
                    "has_verifier_reward": [1, 1, 0, 0],
                    "reward_sources": ["verifier", "verifier", "heuristic", "heuristic"],
                    "reward_evidence_kinds": ["verifier", "verifier", "heuristic", "heuristic"],
                    "reward_truth_owners": ["verifier", "verifier", "heuristic", "heuristic"],
                },
            },
            {
                "prompt_token_ids": [[1, 4], [1, 4], [1, 5], [1, 5]],
                "response_ids": [[7], [8], [9], [10]],
                "rewards": [[0.0], [1.0], [0.0], [1.0]],
                "loss_masks": [[1], [1], [1], [1]],
                "rollout_logprobs": None,
                "stop_reasons": ["stop", "stop", "stop", "stop"],
                "rollout_metrics": {
                    "n_success": 4,
                    "n_with_verifier_reward": 2,
                    "has_verifier_reward": [0, 0, 1, 1],
                    "reward_sources": ["heuristic", "heuristic", "verifier", "verifier"],
                    "reward_evidence_kinds": ["heuristic", "heuristic", "verifier", "verifier"],
                    "reward_truth_owners": ["heuristic", "heuristic", "verifier", "verifier"],
                },
            },
        ]

        class _FilterGenerator:
            def __init__(self):
                self.calls = 0

            async def generate(self, input_batch):
                del input_batch
                out = batches[self.calls]
                self.calls += 1
                return out

        filter_trainer.generator = _FilterGenerator()
        out1 = asyncio.run(filter_trainer.generate({"prompts": ["A0", "A1", "B0", "B1"]}))
        out1, uids1, keep1 = filter_trainer.handle_dynamic_sampling(out1, ["A", "A", "B", "B"])
        assert keep1 is True

        out2 = asyncio.run(filter_trainer.generate({"prompts": ["C0", "C1", "D0", "D1"]}))
        final_out, final_uids, keep2 = filter_trainer.handle_dynamic_sampling(
            out2,
            ["C", "C", "D", "D"],
        )
        assert keep2 is False
        assert final_uids == ["A", "A", "C", "C"]

        ti_filter = filter_trainer.convert_to_training_input(final_out, final_uids)
        ti_filter["advantages"] = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float32)
        ti_filter["action_log_probs"] = torch.zeros((4, 1), dtype=torch.float32)
        ti_filter = filter_trainer._normalize_advantages(ti_filter)

        assert ti_filter.metadata["rollout_metrics"]["has_verifier_reward"] == [1, 1, 0, 0]
        assert ti_filter.metadata["rollout_metrics"]["n_with_verifier_reward"] == 2
        assert ti_filter.metadata["rollout_metrics"]["reward_truth_owners"] == [
            "verifier",
            "verifier",
            "heuristic",
            "heuristic",
        ]
        assert ti_filter["lineage_loss_weights"].tolist() == pytest.approx([1.5, 1.5, 1.0, 1.0])
