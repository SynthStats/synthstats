"""FlowRL loss tests (arXiv:2509.15207).

Tests cover importance weight clamping, behavior without old_log_probs,
numerical stability, and gradient flow to logZ.
"""

import pytest
import torch

from synthstats.train.objectives.losses import compute_flowrl_loss


class TestImportanceWeightClamping:
    """Large log_ratio should be capped at [-5, 5]."""

    def test_clamped_weights_are_bounded(self):
        B, T = 4, 10
        torch.manual_seed(42)
        log_probs = torch.randn(B, T)
        log_rewards = torch.randn(B)
        mask = torch.ones(B, T)
        logZ = torch.tensor(0.0)

        # old_log_probs very different from current -> large log_ratio
        old_log_probs = log_probs + 100.0

        loss, metrics = compute_flowrl_loss(
            log_probs,
            log_rewards,
            mask,
            logZ,
            old_log_probs=old_log_probs,
            use_importance_weights=True,
        )

        assert torch.isfinite(loss)
        # max IS weight = exp(5) ≈ 148.4
        assert metrics["mean_is_weight"] <= 150.0

    def test_identical_old_log_probs_unit_weight(self):
        """When old_log_probs == log_probs, IS weights should be 1.0."""
        B, T = 4, 10
        torch.manual_seed(42)
        log_probs = torch.randn(B, T)
        log_rewards = torch.randn(B)
        mask = torch.ones(B, T)
        logZ = torch.tensor(0.0)

        loss, metrics = compute_flowrl_loss(
            log_probs,
            log_rewards,
            mask,
            logZ,
            old_log_probs=log_probs.clone(),
            use_importance_weights=True,
        )

        assert metrics["mean_is_weight"] == pytest.approx(1.0, abs=1e-5)


class TestWithoutImportanceWeights:
    """use_importance_weights=False or old_log_probs=None."""

    def test_no_old_log_probs(self):
        B, T = 4, 10
        torch.manual_seed(42)
        log_probs = torch.randn(B, T)
        log_rewards = torch.randn(B)
        mask = torch.ones(B, T)
        logZ = torch.tensor(0.0)

        loss, metrics = compute_flowrl_loss(
            log_probs,
            log_rewards,
            mask,
            logZ,
            old_log_probs=None,
            use_importance_weights=False,
        )

        assert torch.isfinite(loss)
        assert metrics["mean_is_weight"] == pytest.approx(1.0, abs=1e-5)

    def test_use_importance_weights_true_but_no_old_probs_falls_back(self):
        """Should fall back to unit weights when old_log_probs is None."""
        B, T = 2, 5
        log_probs = torch.randn(B, T)
        log_rewards = torch.randn(B)
        mask = torch.ones(B, T)
        logZ = torch.tensor(0.0)

        loss, metrics = compute_flowrl_loss(
            log_probs,
            log_rewards,
            mask,
            logZ,
            old_log_probs=None,
            use_importance_weights=True,
        )
        assert metrics["mean_is_weight"] == pytest.approx(1.0, abs=1e-5)

    def test_disabled_importance_weights_ignores_old_probs(self):
        """use_importance_weights=False should ignore old_log_probs entirely."""
        B, T = 4, 10
        torch.manual_seed(42)
        log_probs = torch.randn(B, T)
        log_rewards = torch.randn(B)
        mask = torch.ones(B, T)
        logZ = torch.tensor(0.0)

        old_log_probs = log_probs + 50.0  # very different

        loss_no_is, m_no_is = compute_flowrl_loss(
            log_probs,
            log_rewards,
            mask,
            logZ,
            old_log_probs=old_log_probs,
            use_importance_weights=False,
        )
        loss_no_old, m_no_old = compute_flowrl_loss(
            log_probs,
            log_rewards,
            mask,
            logZ,
            old_log_probs=None,
            use_importance_weights=False,
        )

        assert loss_no_is.item() == pytest.approx(loss_no_old.item(), abs=1e-5)


class TestFlowRLFiniteOutput:
    """Extreme inputs should produce finite output."""

    def test_extreme_log_probs(self):
        B, T = 2, 5
        log_probs = torch.full((B, T), -500.0)
        log_rewards = torch.tensor([500.0, -500.0])
        mask = torch.ones(B, T)
        logZ = torch.tensor(0.0)

        loss, _ = compute_flowrl_loss(
            log_probs,
            log_rewards,
            mask,
            logZ,
            max_residual=50.0,
        )

        assert torch.isfinite(loss)

    def test_all_masked_produces_finite(self):
        B, T = 2, 5
        log_probs = torch.randn(B, T)
        log_rewards = torch.randn(B)
        mask = torch.zeros(B, T)
        logZ = torch.tensor(0.0)

        loss, _ = compute_flowrl_loss(
            log_probs,
            log_rewards,
            mask,
            logZ,
        )

        assert torch.isfinite(loss)

    def test_zero_reward_temp_raises(self):
        with pytest.raises(ValueError, match="reward_temp must be positive"):
            compute_flowrl_loss(
                torch.randn(2, 5),
                torch.randn(2),
                torch.ones(2, 5),
                torch.tensor(0.0),
                reward_temp=0.0,
            )


class TestFlowRLGradientFlow:
    """Gradient flows to logZ and log_probs."""

    def test_gradient_to_logZ(self):
        B, T = 4, 10
        torch.manual_seed(42)
        log_probs = torch.randn(B, T, requires_grad=True)
        log_rewards = torch.randn(B)
        mask = torch.ones(B, T)
        logZ = torch.tensor(1.0, requires_grad=True)

        loss, _ = compute_flowrl_loss(
            log_probs,
            log_rewards,
            mask,
            logZ,
        )
        loss.backward()

        assert logZ.grad is not None
        assert logZ.grad.item() != 0.0

    def test_gradient_to_log_probs(self):
        B, T = 4, 10
        torch.manual_seed(42)
        log_probs = torch.randn(B, T, requires_grad=True)
        log_rewards = torch.randn(B)
        mask = torch.ones(B, T)
        logZ = torch.tensor(0.0)

        loss, _ = compute_flowrl_loss(
            log_probs,
            log_rewards,
            mask,
            logZ,
        )
        loss.backward()

        assert log_probs.grad is not None
        assert not torch.all(log_probs.grad == 0)

    def test_importance_weights_detached(self):
        """IS weights should be detached (no gradient through them)."""
        B, T = 4, 10
        torch.manual_seed(42)
        log_probs = torch.randn(B, T, requires_grad=True)
        log_rewards = torch.randn(B)
        mask = torch.ones(B, T)
        logZ = torch.tensor(0.0)
        old_log_probs = torch.randn(B, T)

        loss, _ = compute_flowrl_loss(
            log_probs,
            log_rewards,
            mask,
            logZ,
            old_log_probs=old_log_probs,
            use_importance_weights=True,
        )
        loss.backward()

        # gradients should still flow to log_probs (through residual, not weights)
        assert log_probs.grad is not None


class TestFlowRLRefLogProbs:
    """KL regularization via ref_log_probs."""

    def test_ref_log_probs_affect_loss(self):
        torch.manual_seed(42)
        B, T = 4, 10
        log_probs = torch.randn(B, T)
        log_rewards = torch.randn(B)
        mask = torch.ones(B, T)
        logZ = torch.tensor(0.0)

        loss_no_ref, _ = compute_flowrl_loss(
            log_probs,
            log_rewards,
            mask,
            logZ,
            ref_log_probs=None,
        )
        loss_with_ref, _ = compute_flowrl_loss(
            log_probs,
            log_rewards,
            mask,
            logZ,
            ref_log_probs=torch.randn(B, T) * 2,
        )

        assert loss_no_ref.item() != pytest.approx(loss_with_ref.item(), abs=0.01)

    def test_length_normalization_with_ref(self):
        torch.manual_seed(42)
        B, T = 4, 10
        log_probs = torch.randn(B, T)
        log_rewards = torch.randn(B)
        mask = torch.ones(B, T)
        logZ = torch.tensor(0.0)
        ref = torch.randn(B, T)

        loss_normed, _ = compute_flowrl_loss(
            log_probs,
            log_rewards,
            mask,
            logZ,
            ref_log_probs=ref,
            length_normalize=True,
        )
        loss_raw, _ = compute_flowrl_loss(
            log_probs,
            log_rewards,
            mask,
            logZ,
            ref_log_probs=ref,
            length_normalize=False,
        )

        assert loss_normed.item() != pytest.approx(loss_raw.item(), abs=0.01)
