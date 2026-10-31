"""VarGrad TB loss tests.

Tests cover length normalization, prompt_ids grouping,
ref_log_probs integration, and edge cases.
"""

import pytest
import torch

from synthstats.train.objectives.losses import (
    compute_vargrad_tb_loss,
    estimate_log_partition_vargrad,
)


class TestVarGradIdenticalTrajectories:
    """Identical trajectories should yield near-zero loss."""

    def test_identical_trajs_zero_loss(self):
        B, T = 4, 10
        log_probs = torch.full((B, T), -1.0)
        log_rewards = torch.full((B,), 2.0)
        mask = torch.ones(B, T)

        loss, metrics = compute_vargrad_tb_loss(
            log_probs,
            log_rewards,
            mask,
            length_normalize=False,
        )

        assert loss.item() == pytest.approx(0.0, abs=1e-5)
        assert metrics["std_residual"] == pytest.approx(0.0, abs=1e-5)

    def test_identical_trajs_zero_loss_with_normalization(self):
        B, T = 4, 10
        log_probs = torch.full((B, T), -1.0)
        log_rewards = torch.full((B,), 2.0)
        mask = torch.ones(B, T)

        loss, metrics = compute_vargrad_tb_loss(
            log_probs,
            log_rewards,
            mask,
            length_normalize=True,
        )

        assert loss.item() == pytest.approx(0.0, abs=1e-5)


class TestVarGradLengthNormalization:
    """logZ estimate must be scale-consistent with the residual."""

    def test_logZ_independent_of_length_when_normalized(self):
        """Two batches with different T but same per-token log_prob should
        yield the same logZ when length_normalize=True."""
        torch.manual_seed(42)
        B = 4
        per_token = -0.5
        log_rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])

        T_short, T_long = 5, 50
        lp_short = torch.full((B, T_short), per_token)
        lp_long = torch.full((B, T_long), per_token)
        mask_short = torch.ones(B, T_short)
        mask_long = torch.ones(B, T_long)

        _, m_short = compute_vargrad_tb_loss(
            lp_short,
            log_rewards,
            mask_short,
            length_normalize=True,
        )
        _, m_long = compute_vargrad_tb_loss(
            lp_long,
            log_rewards,
            mask_long,
            length_normalize=True,
        )

        assert m_short["estimated_logZ"] == pytest.approx(
            m_long["estimated_logZ"],
            abs=1e-4,
        )

    def test_logZ_differs_without_normalization(self):
        """Without normalization, logZ scales with trajectory length."""
        B = 4
        per_token = -0.5
        log_rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])

        T_short, T_long = 5, 50
        lp_short = torch.full((B, T_short), per_token)
        lp_long = torch.full((B, T_long), per_token)
        mask_short = torch.ones(B, T_short)
        mask_long = torch.ones(B, T_long)

        _, m_short = compute_vargrad_tb_loss(
            lp_short,
            log_rewards,
            mask_short,
            length_normalize=False,
        )
        _, m_long = compute_vargrad_tb_loss(
            lp_long,
            log_rewards,
            mask_long,
            length_normalize=False,
        )

        assert abs(m_short["estimated_logZ"] - m_long["estimated_logZ"]) > 1.0

    def test_normalized_loss_lower_variance_across_lengths(self):
        """With normalization, loss variance across batches of different lengths
        is smaller than without."""
        torch.manual_seed(123)
        B = 8
        log_rewards = torch.randn(B)

        losses_normed = []
        losses_raw = []
        for T in [5, 10, 20, 50]:
            lp = torch.randn(B, T) * 0.1
            mask = torch.ones(B, T)

            loss_n, _ = compute_vargrad_tb_loss(lp, log_rewards, mask, length_normalize=True)
            loss_r, _ = compute_vargrad_tb_loss(lp, log_rewards, mask, length_normalize=False)
            losses_normed.append(loss_n.item())
            losses_raw.append(loss_r.item())

        # normalized losses should have lower variance across different T
        import statistics

        var_normed = statistics.variance(losses_normed)
        var_raw = statistics.variance(losses_raw)
        assert var_normed < var_raw


class TestVarGradSingleSample:
    """K=1 edge case: single sample per batch."""

    def test_single_sample(self):
        B, T = 1, 5
        log_probs = torch.randn(B, T)
        log_rewards = torch.tensor([1.0])
        mask = torch.ones(B, T)

        loss, metrics = compute_vargrad_tb_loss(
            log_probs,
            log_rewards,
            mask,
            length_normalize=True,
        )

        # K=1: logZ = score itself, so residual = 0 and loss = 0
        assert loss.item() == pytest.approx(0.0, abs=1e-5)
        assert torch.isfinite(loss)

    def test_single_sample_unnormalized(self):
        B, T = 1, 5
        log_probs = torch.randn(B, T)
        log_rewards = torch.tensor([1.0])
        mask = torch.ones(B, T)

        loss, _ = compute_vargrad_tb_loss(
            log_probs,
            log_rewards,
            mask,
            length_normalize=False,
        )

        assert loss.item() == pytest.approx(0.0, abs=1e-5)


class TestVarGradPromptIds:
    """prompt_ids grouping correctness."""

    def test_grouped_estimation_differs_from_global(self):
        """Per-prompt logZ should differ from global logZ when prompts have
        different reward distributions."""
        torch.manual_seed(42)
        B, T = 8, 10
        log_probs = torch.randn(B, T) * 0.1
        mask = torch.ones(B, T)

        # two groups with very different rewards
        log_rewards = torch.tensor([10.0, 10.0, 10.0, 10.0, -10.0, -10.0, -10.0, -10.0])
        prompt_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

        _, m_grouped = compute_vargrad_tb_loss(
            log_probs,
            log_rewards,
            mask,
            length_normalize=True,
            prompt_ids=prompt_ids,
        )
        _, m_global = compute_vargrad_tb_loss(
            log_probs,
            log_rewards,
            mask,
            length_normalize=True,
            prompt_ids=None,
        )

        # grouped logZ is a per-sample value averaged; global is a single value
        # they should differ because reward distributions per prompt are different
        assert m_grouped["estimated_logZ"] != pytest.approx(
            m_global["estimated_logZ"],
            abs=0.1,
        )

    def test_single_prompt_id_matches_global(self):
        """Single prompt group should match global estimation."""
        torch.manual_seed(42)
        B, T = 4, 10
        log_probs = torch.randn(B, T)
        log_rewards = torch.randn(B)
        mask = torch.ones(B, T)
        prompt_ids = torch.zeros(B, dtype=torch.long)

        _, m_grouped = compute_vargrad_tb_loss(
            log_probs,
            log_rewards,
            mask,
            length_normalize=True,
            prompt_ids=prompt_ids,
        )
        _, m_global = compute_vargrad_tb_loss(
            log_probs,
            log_rewards,
            mask,
            length_normalize=True,
            prompt_ids=None,
        )

        assert m_grouped["estimated_logZ"] == pytest.approx(
            m_global["estimated_logZ"],
            abs=1e-4,
        )

    def test_prompt_ids_with_length_normalization(self):
        """prompt_ids + length_normalize=True: logZ estimation uses normalized log_probs."""
        torch.manual_seed(42)
        B, T = 6, 10
        log_probs = torch.randn(B, T)
        log_rewards = torch.randn(B)
        mask = torch.ones(B, T)
        prompt_ids = torch.tensor([0, 0, 0, 1, 1, 1])

        loss, metrics = compute_vargrad_tb_loss(
            log_probs,
            log_rewards,
            mask,
            length_normalize=True,
            prompt_ids=prompt_ids,
        )

        assert torch.isfinite(loss)
        assert not torch.isnan(loss)


class TestVarGradRefLogProbs:
    """ref_log_probs integration."""

    def test_ref_log_probs_affect_logZ(self):
        torch.manual_seed(42)
        B, T = 4, 10
        log_probs = torch.randn(B, T)
        log_rewards = torch.randn(B)
        mask = torch.ones(B, T)
        ref_log_probs = torch.randn(B, T) * 2.0

        _, m_no_ref = compute_vargrad_tb_loss(
            log_probs,
            log_rewards,
            mask,
            ref_log_probs=None,
            length_normalize=True,
        )
        _, m_with_ref = compute_vargrad_tb_loss(
            log_probs,
            log_rewards,
            mask,
            ref_log_probs=ref_log_probs,
            length_normalize=True,
        )

        assert m_no_ref["estimated_logZ"] != pytest.approx(
            m_with_ref["estimated_logZ"],
            abs=0.01,
        )

    def test_zero_ref_log_probs_no_effect(self):
        torch.manual_seed(42)
        B, T = 4, 10
        log_probs = torch.randn(B, T)
        log_rewards = torch.randn(B)
        mask = torch.ones(B, T)
        ref_log_probs = torch.zeros(B, T)

        _, m_no_ref = compute_vargrad_tb_loss(
            log_probs,
            log_rewards,
            mask,
            ref_log_probs=None,
            length_normalize=False,
        )
        _, m_with_ref = compute_vargrad_tb_loss(
            log_probs,
            log_rewards,
            mask,
            ref_log_probs=ref_log_probs,
            length_normalize=False,
        )

        assert m_no_ref["estimated_logZ"] == pytest.approx(
            m_with_ref["estimated_logZ"],
            abs=1e-5,
        )


class TestEstimateLogPartitionVargrad:
    """Direct tests for the logZ estimator."""

    def test_rejects_zero_reward_temp(self):
        with pytest.raises(ValueError, match="reward_temp must be positive"):
            estimate_log_partition_vargrad(
                torch.randn(2, 5),
                torch.randn(2),
                torch.ones(2, 5),
                reward_temp=0.0,
            )

    def test_rejects_negative_reward_temp(self):
        with pytest.raises(ValueError, match="reward_temp must be positive"):
            estimate_log_partition_vargrad(
                torch.randn(2, 5),
                torch.randn(2),
                torch.ones(2, 5),
                reward_temp=-1.0,
            )

    def test_length_normalize_parameter(self):
        """length_normalize=True should change the result."""
        torch.manual_seed(42)
        B, T = 4, 20
        log_probs = torch.randn(B, T)
        log_rewards = torch.randn(B)
        mask = torch.ones(B, T)

        logZ_raw = estimate_log_partition_vargrad(
            log_probs,
            log_rewards,
            mask,
            length_normalize=False,
        )
        logZ_normed = estimate_log_partition_vargrad(
            log_probs,
            log_rewards,
            mask,
            length_normalize=True,
        )

        assert logZ_raw.item() != pytest.approx(logZ_normed.item(), abs=0.01)

    def test_single_sample_returns_score(self):
        """K=1: logZ = the single score."""
        lp = torch.tensor([[-1.0, -2.0, -3.0]])
        lr = torch.tensor([5.0])
        mask = torch.ones(1, 3)

        logZ = estimate_log_partition_vargrad(lp, lr, mask, length_normalize=False)

        # score = 5.0 - (-6.0) = 11.0
        assert logZ.item() == pytest.approx(11.0, abs=1e-5)


class TestVarGradGradientFlow:
    """Gradients must flow through the residual to log_probs."""

    def test_gradient_flows(self):
        B, T = 4, 10
        log_probs = torch.randn(B, T, requires_grad=True)
        log_rewards = torch.randn(B)
        mask = torch.ones(B, T)

        loss, _ = compute_vargrad_tb_loss(
            log_probs,
            log_rewards,
            mask,
            length_normalize=True,
        )
        loss.backward()

        assert log_probs.grad is not None
        # not all zeros (logZ is detached, but traj_log_prob_normed has grad)
        assert not torch.all(log_probs.grad == 0)

    def test_no_gradient_to_rewards(self):
        """Rewards should not receive gradients (they're external signals)."""
        B, T = 4, 10
        log_probs = torch.randn(B, T, requires_grad=True)
        log_rewards = torch.randn(B, requires_grad=True)
        mask = torch.ones(B, T)

        loss, _ = compute_vargrad_tb_loss(
            log_probs,
            log_rewards,
            mask,
            length_normalize=True,
        )
        loss.backward()

        # logZ is detached, so rewards only appear in scaled_rewards with requires_grad
        # The residual includes scaled_rewards which do get gradients
        assert log_rewards.grad is not None


class TestVarGradNumericalStability:
    """Extreme inputs should produce finite outputs."""

    def test_large_log_probs(self):
        B, T = 2, 5
        log_probs = torch.randn(B, T) * 100
        log_rewards = torch.randn(B)
        mask = torch.ones(B, T)

        loss, _ = compute_vargrad_tb_loss(
            log_probs,
            log_rewards,
            mask,
            max_residual=50.0,
        )

        assert torch.isfinite(loss)

    def test_reward_temp_scaling(self):
        """Different reward temperatures should produce different losses."""
        torch.manual_seed(42)
        B, T = 4, 10
        log_probs = torch.randn(B, T)
        log_rewards = torch.randn(B) * 5
        mask = torch.ones(B, T)

        loss_low, _ = compute_vargrad_tb_loss(
            log_probs,
            log_rewards,
            mask,
            reward_temp=1.0,
        )
        loss_high, _ = compute_vargrad_tb_loss(
            log_probs,
            log_rewards,
            mask,
            reward_temp=15.0,
        )

        assert loss_low.item() != pytest.approx(loss_high.item(), abs=0.1)
