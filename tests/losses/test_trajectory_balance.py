"""Tests for Trajectory Balance loss and tb_identity advantage estimator."""

import pytest
import torch

from synthstats.integrations.skyrl.registry import SKYRL_AVAILABLE
from synthstats.train.objectives.trajectory_balance import (
    compute_modified_subtb_loss,
    compute_tb_identity_advantage,
    compute_trajectory_balance_loss,
)


class TestTBIdentityAdvantageEstimator:
    def test_sums_rewards_per_trajectory(self):
        token_rewards = torch.tensor(
            [
                [0.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 1.5, 0.0],
            ]
        )
        mask = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 0.0],
            ]
        )

        advantages, returns = compute_tb_identity_advantage(token_rewards, mask)

        assert advantages.shape == token_rewards.shape
        assert returns.shape == token_rewards.shape

        assert torch.allclose(advantages[0], torch.tensor([2.0, 2.0, 2.0, 2.0]))
        assert torch.allclose(advantages[1], torch.tensor([1.5, 1.5, 1.5, 0.0]))

    def test_respects_mask(self):
        token_rewards = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])

        advantages, _ = compute_tb_identity_advantage(token_rewards, mask)

        assert advantages[0, 0] == 3.0
        assert advantages[0, 1] == 3.0
        assert advantages[0, 2] == 0.0

    def test_returns_same_as_advantages(self):
        token_rewards = torch.randn(4, 10)
        mask = torch.ones(4, 10)

        advantages, returns = compute_tb_identity_advantage(token_rewards, mask)

        assert torch.equal(advantages, returns)

    def test_no_gradients(self):
        token_rewards = torch.randn(2, 5, requires_grad=True)
        mask = torch.ones(2, 5)

        advantages, returns = compute_tb_identity_advantage(token_rewards, mask)

        assert not advantages.requires_grad
        assert not returns.requires_grad

    def test_ignores_kwargs(self):
        token_rewards = torch.tensor([[1.0, 2.0]])
        mask = torch.ones(1, 2)

        advantages, returns = compute_tb_identity_advantage(
            token_rewards,
            mask,
            gamma=0.99,
            lambd=0.95,
            values=torch.randn(1, 2),
            index=[0],
        )

        assert advantages.shape == token_rewards.shape


class TestTrajectoryBalanceLoss:
    def test_computes_squared_residual(self):
        log_probs = torch.tensor(
            [
                [-1.0, -1.0, -1.0],
                [-0.5, -0.5, -0.5],
            ]
        )
        advantages = torch.tensor(
            [
                [2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0],
            ]
        )
        loss_mask = torch.ones(2, 3)

        config = {"logZ": 0.5, "tb_max_residual": 100.0}

        loss, clip_ratio = compute_trajectory_balance_loss(
            log_probs, log_probs, advantages, config, loss_mask
        )

        # logZ=0.5 + log_pi=-3 - log_R=2.0 = -4.5, sq=20.25;
        # logZ=0.5 - 1.5 - 1.0 = -2.0, sq=4.0; mean=12.125
        assert torch.isclose(loss, torch.tensor(12.125), atol=0.01)
        assert isinstance(clip_ratio, dict)
        assert clip_ratio["clip_ratio"] == 0.0

    def test_requires_logZ_in_config(self):
        log_probs = torch.randn(2, 5)
        advantages = torch.randn(2, 5)
        config = {"tb_max_residual": 100.0}

        with pytest.raises(RuntimeError, match="logZ not found"):
            compute_trajectory_balance_loss(log_probs, log_probs, advantages, config)

    def test_preserves_large_residual_and_gradient(self):
        log_probs = torch.tensor([[-1000.0]], requires_grad=True)
        advantages = torch.tensor([[1000.0]])
        config = {"logZ": 0.0, "tb_max_residual": 50.0}

        loss, _ = compute_trajectory_balance_loss(log_probs, log_probs, advantages, config)

        assert loss.item() == pytest.approx(4_000_000.0)
        loss.backward()
        assert log_probs.grad.item() == pytest.approx(-4000.0)

    def test_handles_nan_in_rewards(self):
        log_probs = torch.tensor([[-1.0, -1.0]])
        advantages = torch.tensor([[float("nan"), float("nan")]])
        config = {"logZ": 0.0, "tb_max_residual": 100.0}

        with pytest.raises(ValueError, match="finite log rewards"):
            compute_trajectory_balance_loss(log_probs, log_probs, advantages, config)

    def test_handles_inf_in_rewards(self):
        log_probs = torch.tensor([[-1.0, -1.0]])
        advantages = torch.tensor([[float("inf"), float("inf")]])
        config = {"logZ": 0.0, "tb_max_residual": 100.0}

        with pytest.raises(ValueError, match="finite log rewards"):
            compute_trajectory_balance_loss(log_probs, log_probs, advantages, config)

    def test_works_without_mask(self):
        log_probs = torch.randn(2, 5)
        advantages = torch.randn(2, 5)
        config = {"logZ": 0.0}

        loss, clip_ratio = compute_trajectory_balance_loss(
            log_probs, log_probs, advantages, config, loss_mask=None
        )

        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_respects_mask(self):
        log_probs = torch.tensor([[-1.0, -1.0, -999.0]])
        advantages = torch.tensor([[1.0, 1.0, 1.0]])
        loss_mask = torch.tensor([[1.0, 1.0, 0.0]])
        config = {"logZ": 0.0}

        loss, _ = compute_trajectory_balance_loss(
            log_probs, log_probs, advantages, config, loss_mask
        )

        assert torch.isclose(loss, torch.tensor(9.0), atol=0.1)

    def test_applies_lineage_loss_weight(self):
        log_probs = torch.tensor([[-1.0, -1.0]])
        advantages = torch.tensor([[2.0, 2.0]])
        loss_mask = torch.ones(1, 2)

        unweighted_loss, _ = compute_trajectory_balance_loss(
            log_probs, log_probs, advantages, {"logZ": 0.0}, loss_mask
        )
        weighted_loss, metrics = compute_trajectory_balance_loss(
            log_probs,
            log_probs,
            advantages,
            {"logZ": 0.0, "lineage_loss_weight": 1.5},
            loss_mask,
        )

        assert torch.isclose(weighted_loss, unweighted_loss * 1.5)
        assert metrics["lineage_loss_weight"] == pytest.approx(1.5)

    def test_applies_per_sample_lineage_loss_weights(self):
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

        weighted_loss, metrics = compute_trajectory_balance_loss(
            log_probs,
            log_probs,
            advantages,
            {"logZ": 0.0, "lineage_loss_weights": [2.0, 1.0]},
            loss_mask,
        )

        expected = torch.tensor((2.0 * 9.0 + 1.0 * 25.0) / 2.0)
        assert torch.isclose(weighted_loss, expected)
        assert metrics["lineage_loss_weight"] == pytest.approx(1.5)
        assert metrics["lineage_loss_weight_min"] == pytest.approx(1.0)
        assert metrics["lineage_loss_weight_max"] == pytest.approx(2.0)

    def test_filters_per_sample_lineage_weights_with_masked_rows(self):
        log_probs = torch.tensor(
            [
                [-5.0, -5.0],
                [-1.0, -1.0],
            ],
            requires_grad=True,
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

        weighted_loss, metrics = compute_trajectory_balance_loss(
            log_probs,
            log_probs.detach(),
            advantages,
            {"logZ": 0.0, "lineage_loss_weights": [10.0, 1.0]},
            loss_mask,
        )

        assert torch.isclose(weighted_loss, torch.tensor(9.0))
        assert metrics["lineage_loss_weight"] == pytest.approx(1.0)
        assert metrics["lineage_loss_weight_min"] == pytest.approx(1.0)
        assert metrics["lineage_loss_weight_max"] == pytest.approx(1.0)

    def test_modified_subtb_loss_applies_per_sample_lineage_weights(self):
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

        weighted_loss, metrics = compute_modified_subtb_loss(
            log_probs,
            log_probs,
            advantages,
            {
                "_eos_logprobs": eos_logprobs,
                "subtb_lambda": 0.9,
                "lineage_loss_weights": [2.0, 1.0],
            },
            loss_mask,
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

    def test_all_masked_loss_is_differentiable(self):
        log_probs = torch.randn(2, 4, requires_grad=True)
        advantages = torch.zeros(2, 4)
        loss_mask = torch.zeros(2, 4)
        config = {"logZ": 0.0, "tb_max_residual": 100.0}

        loss, _ = compute_trajectory_balance_loss(
            log_probs, log_probs.detach(), advantages, config, loss_mask
        )

        assert loss.shape == ()
        assert loss.requires_grad

        loss.backward()
        assert log_probs.grad is not None
        assert torch.allclose(log_probs.grad, torch.zeros_like(log_probs.grad))

    def test_config_types(self):
        log_probs = torch.randn(2, 5)
        advantages = torch.randn(2, 5)

        loss1, _ = compute_trajectory_balance_loss(log_probs, log_probs, advantages, {"logZ": 0.5})

        from omegaconf import DictConfig

        cfg = DictConfig({"logZ": 0.5})
        loss2, _ = compute_trajectory_balance_loss(log_probs, log_probs, advantages, cfg)

        assert torch.isclose(loss1, loss2)


class TestSkyRLRegistration:
    @pytest.mark.skipif(not SKYRL_AVAILABLE, reason="SkyRL not available")
    def test_tb_identity_registered(self, monkeypatch):
        from skyrl.backends.skyrl_train.utils.ppo_utils import AdvantageEstimatorRegistry

        from synthstats.integrations.skyrl import registry

        monkeypatch.setattr(registry, "_REGISTERED", False)
        registry.register_all()
        available = AdvantageEstimatorRegistry.list_available()
        assert "tb_identity" in available

    @pytest.mark.skipif(not SKYRL_AVAILABLE, reason="SkyRL not available")
    def test_trajectory_balance_registered(self):
        from skyrl.backends.skyrl_train.utils.ppo_utils import PolicyLossRegistry

        from synthstats.integrations.skyrl.registry import register_all

        register_all()
        available = PolicyLossRegistry.list_available()
        assert "trajectory_balance" in available

    @pytest.mark.skipif(not SKYRL_AVAILABLE, reason="SkyRL not available")
    def test_can_retrieve_tb_identity(self):
        from skyrl.backends.skyrl_train.utils.ppo_utils import AdvantageEstimatorRegistry

        estimator = AdvantageEstimatorRegistry.get("tb_identity")
        assert callable(estimator)

        token_rewards = torch.randn(2, 5)
        mask = torch.ones(2, 5)
        advantages, returns = estimator(token_rewards, mask)
        assert advantages.shape == token_rewards.shape

    @pytest.mark.skipif(not SKYRL_AVAILABLE, reason="SkyRL not available")
    def test_can_retrieve_trajectory_balance(self):
        from skyrl.backends.skyrl_train.utils.ppo_utils import PolicyLossRegistry

        from synthstats.integrations.skyrl.registry import register_all

        register_all()
        loss_fn = PolicyLossRegistry.get("trajectory_balance")
        assert callable(loss_fn)

        log_probs = torch.randn(2, 5)
        advantages = torch.randn(2, 5)
        config = {"logZ": 0.0}
        loss, clip = loss_fn(log_probs, log_probs, advantages, config)
        assert loss.shape == ()


class TestTBTrainerIntegration:
    def test_logZ_module(self):
        from synthstats.integrations.skyrl.trainer import LogZModule

        module = LogZModule(init_value=1.5)

        assert module.logZ.item() == 1.5
        assert module.logZ.requires_grad

    def test_trainer_inject_logZ(self):
        from synthstats.train.gfn_core import GFlowNetTrainerCore

        trainer = GFlowNetTrainerCore()
        trainer.init_gfn(logZ_init=2.5)
        config: dict[str, dict[str, float]] = {"algorithm": {}}

        trainer.inject_logZ_into_config(config)

        assert config["algorithm"]["logZ"] == 2.5

    def test_trainer_inject_logZ_omegaconf(self):
        from omegaconf import DictConfig

        from synthstats.train.gfn_core import GFlowNetTrainerCore

        trainer = GFlowNetTrainerCore()
        trainer.init_gfn(logZ_init=2.5)
        config = DictConfig({"algorithm": {"logZ": 0.0}})

        trainer.inject_logZ_into_config(config)

        assert config.algorithm.logZ == 2.5
        assert hasattr(config, "logZ_module")

    def test_trainer_step(self):
        from synthstats.train.gfn_core import GFlowNetTrainerCore

        trainer = GFlowNetTrainerCore()
        trainer.init_gfn(logZ_init=0.0)

        metrics1 = trainer.gfn_optimizer_step()
        metrics2 = trainer.gfn_optimizer_step()

        assert metrics1["gfn_step"] == 1
        assert metrics2["gfn_step"] == 2


class TestEndToEndTBLoss:
    def test_full_tb_pipeline(self):
        raw_rewards = torch.tensor(
            [
                [0.0, 0.0, 0.0, 10.0],
                [0.0, 0.0, 5.0, 0.0],
            ]
        )
        response_mask = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 0.0],
            ]
        )

        advantages, _ = compute_tb_identity_advantage(raw_rewards, response_mask)
        log_probs = torch.randn(2, 4)
        config = {"logZ": 1.0, "tb_max_residual": 100.0}
        loss, clip = compute_trajectory_balance_loss(
            log_probs, log_probs, advantages, config, response_mask
        )

        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert isinstance(clip, dict)
        assert clip["clip_ratio"] == 0.0

    def test_gradient_propagates_through_loss(self):
        log_probs = torch.randn(2, 5, requires_grad=True)
        advantages = torch.randn(2, 5)
        config = {"logZ": 0.0}

        loss, _ = compute_trajectory_balance_loss(log_probs, log_probs, advantages, config)
        loss.backward()

        assert log_probs.grad is not None
        assert not torch.all(log_probs.grad == 0)

    def test_logZ_gradients_flow_via_tensor(self):
        log_probs = torch.randn(2, 5, requires_grad=True)
        advantages = torch.randn(2, 5)
        logZ_param = torch.tensor(0.5, requires_grad=True)

        class ConfigWithTensor:
            def __init__(self):
                self.logZ = 0.5
                self._logZ_tensor = logZ_param

            def get(self, key, default=None):
                return getattr(self, key, default)

        config = ConfigWithTensor()

        loss, _ = compute_trajectory_balance_loss(log_probs, log_probs, advantages, config)
        loss.backward()

        assert logZ_param.grad is not None
        assert logZ_param.grad != 0.0


class TestRewardTempValidation:
    def test_compute_tb_loss_with_kl_rejects_zero_reward_temp(self):
        from synthstats.train.objectives.losses import compute_tb_loss_with_kl

        log_probs = torch.randn(2, 5)
        log_rewards = torch.randn(2)
        response_mask = torch.ones(2, 5)
        logZ = torch.tensor(0.0)

        with pytest.raises(ValueError, match="reward_temp must be positive"):
            compute_tb_loss_with_kl(log_probs, log_rewards, response_mask, logZ, reward_temp=0.0)

    def test_compute_tb_loss_with_kl_rejects_negative_reward_temp(self):
        from synthstats.train.objectives.losses import compute_tb_loss_with_kl

        log_probs = torch.randn(2, 5)
        log_rewards = torch.randn(2)
        response_mask = torch.ones(2, 5)
        logZ = torch.tensor(0.0)

        with pytest.raises(ValueError, match="reward_temp must be positive"):
            compute_tb_loss_with_kl(log_probs, log_rewards, response_mask, logZ, reward_temp=-1.0)

    def test_compute_flowrl_loss_rejects_zero_reward_temp(self):
        from synthstats.train.objectives.losses import compute_flowrl_loss

        log_probs = torch.randn(2, 5)
        log_rewards = torch.randn(2)
        response_mask = torch.ones(2, 5)
        logZ = torch.tensor(0.0)

        with pytest.raises(ValueError, match="reward_temp must be positive"):
            compute_flowrl_loss(log_probs, log_rewards, response_mask, logZ, reward_temp=0.0)

    def test_compute_flowrl_loss_rejects_negative_reward_temp(self):
        from synthstats.train.objectives.losses import compute_flowrl_loss

        log_probs = torch.randn(2, 5)
        log_rewards = torch.randn(2)
        response_mask = torch.ones(2, 5)
        logZ = torch.tensor(0.0)

        with pytest.raises(ValueError, match="reward_temp must be positive"):
            compute_flowrl_loss(log_probs, log_rewards, response_mask, logZ, reward_temp=-1.0)

    def test_compute_tb_loss_with_kl_accepts_positive_reward_temp(self):
        from synthstats.train.objectives.losses import compute_tb_loss_with_kl

        log_probs = torch.randn(2, 5)
        log_rewards = torch.randn(2)
        response_mask = torch.ones(2, 5)
        logZ = torch.tensor(0.0)

        loss, metrics = compute_tb_loss_with_kl(
            log_probs, log_rewards, response_mask, logZ, reward_temp=1.0
        )
        assert not torch.isnan(loss)

    def test_compute_flowrl_loss_accepts_positive_reward_temp(self):
        from synthstats.train.objectives.losses import compute_flowrl_loss

        log_probs = torch.randn(2, 5)
        log_rewards = torch.randn(2)
        response_mask = torch.ones(2, 5)
        logZ = torch.tensor(0.0)

        loss, metrics = compute_flowrl_loss(
            log_probs, log_rewards, response_mask, logZ, reward_temp=15.0
        )
        assert not torch.isnan(loss)
