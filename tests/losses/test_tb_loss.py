"""Tests for TB (Trajectory Balance) loss."""

import pytest
import torch
import torch.nn as nn


class TestTBLossShape:
    def test_tb_loss_returns_scalar(self):
        from synthstats.train.objectives.losses import tb_loss

        log_probs = torch.tensor([[-0.5, -0.3, -0.2], [-0.4, -0.6, -0.1]])
        loss_mask = torch.ones(2, 3, dtype=torch.bool)
        log_rewards = torch.tensor([0.5, 0.8])
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = tb_loss(log_probs, loss_mask, log_rewards, logZ)

        assert loss.dim() == 0
        assert loss.dtype == torch.float32 or loss.dtype == torch.float64

    def test_tb_loss_batch_size_one(self):
        from synthstats.train.objectives.losses import tb_loss

        log_probs = torch.tensor([[-0.5, -0.3]])
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([0.5])
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = tb_loss(log_probs, loss_mask, log_rewards, logZ)

        assert loss.dim() == 0
        assert not torch.isnan(loss)


class TestTBLossMasking:
    def test_mask_excludes_tokens(self):
        from synthstats.train.objectives.losses import tb_loss

        log_probs = torch.tensor([[-1.0, -1000.0, -1.0]])
        mask_all = torch.ones(1, 3, dtype=torch.bool)
        mask_middle = torch.tensor([[True, False, True]])
        log_rewards = torch.tensor([0.0])
        logZ = nn.Parameter(torch.tensor(0.0))

        loss_all = tb_loss(log_probs, mask_all, log_rewards, logZ)
        loss_masked = tb_loss(log_probs, mask_middle, log_rewards, logZ)

        assert loss_masked < loss_all

    def test_fully_masked_sequence(self):
        """All tokens masked out: loss = (logZ - log_R)^2."""
        from synthstats.train.objectives.losses import tb_loss

        log_probs = torch.tensor([[-1.0, -2.0, -3.0]])  # [1, 3]
        mask_none = torch.zeros(1, 3, dtype=torch.bool)  # all masked
        log_rewards = torch.tensor([1.0])
        logZ = nn.Parameter(torch.tensor(2.0))

        loss = tb_loss(log_probs, mask_none, log_rewards, logZ)

        # (logZ + 0 - log_R)^2 = (2.0 + 0 - 1.0)^2 = 1.0
        expected = torch.tensor(1.0)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_partial_mask(self):
        from synthstats.train.objectives.losses import tb_loss

        log_probs = torch.tensor([[-1.0, -2.0, -3.0]])
        mask = torch.tensor([[True, False, True]])
        log_rewards = torch.tensor([0.0])
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = tb_loss(log_probs, mask, log_rewards, logZ)

        # (0 + (-4) - 0)^2 = 16.0
        assert torch.isclose(loss, torch.tensor(16.0), atol=1e-5)


class TestTBLossGradient:
    def test_gradient_flows_to_logZ(self):
        from synthstats.train.objectives.losses import tb_loss

        log_probs = torch.tensor([[-0.5, -0.3]])
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([0.5])
        logZ = nn.Parameter(torch.tensor(1.0))

        loss = tb_loss(log_probs, loss_mask, log_rewards, logZ)
        loss.backward()

        assert logZ.grad is not None
        assert not torch.isnan(logZ.grad)

    def test_gradient_nonzero_when_unbalanced(self):
        from synthstats.train.objectives.losses import tb_loss

        log_probs = torch.tensor([[-0.5, -0.3]])
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([0.5])
        logZ = nn.Parameter(torch.tensor(10.0))  # way off balance

        loss = tb_loss(log_probs, loss_mask, log_rewards, logZ)
        loss.backward()

        assert logZ.grad != 0.0

    def test_loss_is_differentiable(self):
        from synthstats.train.objectives.losses import tb_loss

        log_probs = torch.tensor([[-0.5, -0.3]], requires_grad=True)
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([0.5])
        logZ = nn.Parameter(torch.tensor(1.0))

        loss = tb_loss(log_probs, loss_mask, log_rewards, logZ)
        loss.backward()

        assert log_probs.grad is not None


class TestTBLossBalance:
    def test_zero_loss_when_balanced(self):
        """logZ + log_pi = log_R implies zero loss."""
        from synthstats.train.objectives.losses import tb_loss

        # logZ=1, sum(log_probs)=-2, log_reward=-1: 1 + (-2) - (-1) = 0
        log_probs = torch.tensor([[-1.0, -1.0]])
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([-1.0])
        logZ = nn.Parameter(torch.tensor(1.0))

        loss = tb_loss(log_probs, loss_mask, log_rewards, logZ)

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_nonzero_loss_when_unbalanced(self):
        from synthstats.train.objectives.losses import tb_loss

        log_probs = torch.tensor([[-1.0, -1.0]])
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([0.0])
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = tb_loss(log_probs, loss_mask, log_rewards, logZ)

        assert loss > 0
        assert torch.isclose(loss, torch.tensor(4.0), atol=1e-5)


class TestTBLossRefPolicy:
    def test_ref_policy_correction_zero_loss(self):
        from synthstats.train.objectives.losses import tb_loss

        log_probs = torch.tensor([[-1.0, -1.0]])
        ref_log_probs = torch.tensor([[-0.5, -0.5]])
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([-1.0])
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = tb_loss(
            log_probs,
            loss_mask,
            log_rewards,
            logZ,
            ref_log_probs=ref_log_probs,
        )

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


class TestTBLossLengthNormalization:
    def test_length_normalization_balances_lengths(self):
        from synthstats.train.objectives.losses import tb_loss

        log_probs = torch.tensor([[-1.0, -1.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
        loss_mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
        log_rewards = torch.tensor([-1.0, -1.0])
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = tb_loss(
            log_probs,
            loss_mask,
            log_rewards,
            logZ,
            normalize_by_length=True,
        )

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


class TestTBLossBatching:
    def test_batch_averaging(self):
        """batch 1: residual=-2 sq=4; batch 2: residual=-4 sq=16; mean=10."""
        from synthstats.train.objectives.losses import tb_loss

        log_probs = torch.tensor([[-1.0, -1.0], [-2.0, -2.0]])
        loss_mask = torch.ones(2, 2, dtype=torch.bool)
        log_rewards = torch.tensor([0.0, 0.0])
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = tb_loss(log_probs, loss_mask, log_rewards, logZ)

        assert torch.isclose(loss, torch.tensor(10.0), atol=1e-5)


class TestTBLossEdgeCases:
    def test_empty_sequence_length(self):
        from synthstats.train.objectives.losses import tb_loss

        log_probs = torch.zeros(2, 0)
        loss_mask = torch.zeros(2, 0, dtype=torch.bool)
        log_rewards = torch.tensor([1.0, 2.0])
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = tb_loss(log_probs, loss_mask, log_rewards, logZ)

        # (0-1)^2 + (0-2)^2 = 5, mean = 2.5
        assert torch.isclose(loss, torch.tensor(2.5), atol=1e-5)

    def test_single_token(self):
        from synthstats.train.objectives.losses import tb_loss

        log_probs = torch.tensor([[-0.5]])
        loss_mask = torch.ones(1, 1, dtype=torch.bool)
        log_rewards = torch.tensor([0.0])
        logZ = nn.Parameter(torch.tensor(0.5))

        loss = tb_loss(log_probs, loss_mask, log_rewards, logZ)

        # (0.5 + (-0.5) - 0)^2 = 0
        expected = torch.tensor(0.0)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_negative_log_rewards(self):
        from synthstats.train.objectives.losses import tb_loss

        log_probs = torch.tensor([[-1.0, -1.0]])
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([-5.0])
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = tb_loss(log_probs, loss_mask, log_rewards, logZ)

        # (0 + (-2) - (-5))^2 = 9
        assert torch.isclose(loss, torch.tensor(9.0), atol=1e-5)

    def test_large_logZ(self):
        from synthstats.train.objectives.losses import tb_loss

        log_probs = torch.tensor([[-1.0, -1.0]])
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([0.0])
        logZ = nn.Parameter(torch.tensor(100.0))

        loss = tb_loss(log_probs, loss_mask, log_rewards, logZ)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_inf_log_rewards_handled(self):
        from synthstats.train.objectives.losses import tb_loss

        log_probs = torch.tensor([[-1.0, -1.0]])
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([float("-inf")])
        logZ = nn.Parameter(torch.tensor(0.0))

        with pytest.raises(ValueError, match="finite log rewards"):
            tb_loss(log_probs, loss_mask, log_rewards, logZ)

    def test_nan_log_rewards_handled(self):
        from synthstats.train.objectives.losses import tb_loss

        log_probs = torch.tensor([[-1.0, -1.0]])
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([float("nan")])
        logZ = nn.Parameter(torch.tensor(0.0))

        with pytest.raises(ValueError, match="finite log rewards"):
            tb_loss(log_probs, loss_mask, log_rewards, logZ)

    def test_large_residual_preserves_squared_objective(self):
        from synthstats.train.objectives.losses import tb_loss

        log_probs = torch.tensor([[-500.0, -500.0]])
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([0.0])
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = tb_loss(log_probs, loss_mask, log_rewards, logZ)

        assert loss == 1000000.0
        loss.backward()
        assert logZ.grad == -2000.0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
