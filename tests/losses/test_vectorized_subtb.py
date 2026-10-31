"""Equivalence tests: loop-based vs vectorized SubTB."""

import pytest
import torch

from synthstats.train.objectives.losses import (
    compute_modified_subtb_core,
    compute_modified_subtb_core_terms,
)


class TestVectorizedEquivalence:
    """Loop and vectorized implementations must produce identical results."""

    @pytest.mark.parametrize("T", [1, 5, 10, 50, 100, 500])
    def test_numerical_equivalence(self, T):
        """Vectorized matches loop to float32 tolerance."""
        B = 4
        torch.manual_seed(42)
        log_probs = torch.randn(B, T)
        eos_logprobs = torch.randn(B, T)
        log_reward = torch.randn(B)
        mask = torch.ones(B, T)

        loss_loop = compute_modified_subtb_core(
            log_probs,
            eos_logprobs,
            log_reward,
            mask,
            subtb_lambda=0.9,
            max_residual=50.0,
            _force_loop=True,
        )

        loss_vec = compute_modified_subtb_core(
            log_probs,
            eos_logprobs,
            log_reward,
            mask,
            subtb_lambda=0.9,
            max_residual=50.0,
            _force_loop=False,
        )

        torch.testing.assert_close(loss_loop, loss_vec, atol=1e-4, rtol=1e-4)

    def test_gradient_equivalence(self):
        """Gradients match between implementations."""
        B, T = 4, 20
        torch.manual_seed(42)

        lp_loop = torch.randn(B, T, requires_grad=True)
        eos_loop = torch.randn(B, T)
        log_r = torch.randn(B)
        mask = torch.ones(B, T)

        # clone for vectorized path
        lp_vec = lp_loop.detach().clone().requires_grad_(True)

        loss_loop = compute_modified_subtb_core(
            lp_loop,
            eos_loop,
            log_r,
            mask,
            subtb_lambda=0.9,
            max_residual=50.0,
            _force_loop=True,
        )
        loss_loop.backward()

        loss_vec = compute_modified_subtb_core(
            lp_vec,
            eos_loop,
            log_r,
            mask,
            subtb_lambda=0.9,
            max_residual=50.0,
            _force_loop=False,
        )
        loss_vec.backward()

        torch.testing.assert_close(lp_loop.grad, lp_vec.grad, atol=1e-4, rtol=1e-4)

    def test_per_row_terms_equivalence(self):
        """Loop and vectorized per-row numerators/denominators must match."""
        B, T = 4, 20
        torch.manual_seed(42)
        log_probs = torch.randn(B, T)
        eos_logprobs = torch.randn(B, T)
        log_reward = torch.randn(B)
        mask = torch.ones(B, T)

        num_loop, den_loop = compute_modified_subtb_core_terms(
            log_probs,
            eos_logprobs,
            log_reward,
            mask,
            subtb_lambda=0.9,
            max_residual=50.0,
            _force_loop=True,
        )
        num_vec, den_vec = compute_modified_subtb_core_terms(
            log_probs,
            eos_logprobs,
            log_reward,
            mask,
            subtb_lambda=0.9,
            max_residual=50.0,
            _force_loop=False,
        )

        torch.testing.assert_close(num_loop, num_vec, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(den_loop, den_vec, atol=1e-4, rtol=1e-4)

    def test_scalar_loss_matches_terms_reduction(self):
        """The scalar API should equal the reduced per-row terms."""
        B, T = 3, 15
        torch.manual_seed(42)
        log_probs = torch.randn(B, T)
        eos_logprobs = torch.randn(B, T)
        log_reward = torch.randn(B)
        mask = torch.ones(B, T)

        loss = compute_modified_subtb_core(
            log_probs,
            eos_logprobs,
            log_reward,
            mask,
            subtb_lambda=0.9,
            max_residual=50.0,
            _force_loop=False,
        )
        numerators, denominators = compute_modified_subtb_core_terms(
            log_probs,
            eos_logprobs,
            log_reward,
            mask,
            subtb_lambda=0.9,
            max_residual=50.0,
            _force_loop=False,
        )

        reduced = numerators.sum() / denominators.sum().clamp(min=1.0)
        torch.testing.assert_close(loss, reduced, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize(
        "mask_pattern",
        [
            [1, 0, 1, 1, 0, 1],  # non-contiguous gaps
            [0, 0, 1, 1, 1, 0],  # leading/trailing zeros
            [1, 0, 0, 0, 0, 1],  # large gap
            [0, 1, 0, 1, 0, 1],  # alternating
        ],
        ids=["gaps", "leading-trailing", "large-gap", "alternating"],
    )
    def test_masking_equivalence(self, mask_pattern):
        """Non-contiguous masks produce identical results."""
        B = 3
        T = len(mask_pattern)
        torch.manual_seed(42)
        log_probs = torch.randn(B, T)
        eos_logprobs = torch.randn(B, T)
        log_reward = torch.randn(B)
        mask = torch.tensor(mask_pattern, dtype=torch.float32).unsqueeze(0).expand(B, T)

        loss_loop = compute_modified_subtb_core(
            log_probs,
            eos_logprobs,
            log_reward,
            mask,
            subtb_lambda=0.9,
            max_residual=50.0,
            _force_loop=True,
        )

        loss_vec = compute_modified_subtb_core(
            log_probs,
            eos_logprobs,
            log_reward,
            mask,
            subtb_lambda=0.9,
            max_residual=50.0,
            _force_loop=False,
        )

        torch.testing.assert_close(loss_loop, loss_vec, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("lam", [0.0, 0.1, 0.5, 0.9, 1.0])
    def test_lambda_weighting_equivalence(self, lam):
        """Lambda decay weights match."""
        B, T = 3, 15
        torch.manual_seed(42)
        log_probs = torch.randn(B, T)
        eos_logprobs = torch.randn(B, T)
        log_reward = torch.randn(B)
        mask = torch.ones(B, T)

        loss_loop = compute_modified_subtb_core(
            log_probs,
            eos_logprobs,
            log_reward,
            mask,
            subtb_lambda=lam,
            max_residual=50.0,
            _force_loop=True,
        )

        loss_vec = compute_modified_subtb_core(
            log_probs,
            eos_logprobs,
            log_reward,
            mask,
            subtb_lambda=lam,
            max_residual=50.0,
            _force_loop=False,
        )

        torch.testing.assert_close(loss_loop, loss_vec, atol=1e-4, rtol=1e-4)

    def test_variable_length_batch(self):
        """Batch items with different valid lengths produce identical results."""
        B, T = 4, 12
        torch.manual_seed(42)
        log_probs = torch.randn(B, T)
        eos_logprobs = torch.randn(B, T)
        log_reward = torch.randn(B)
        # different valid lengths: 12, 8, 4, 1
        mask = torch.zeros(B, T)
        mask[0, :12] = 1
        mask[1, :8] = 1
        mask[2, :4] = 1
        mask[3, :1] = 1

        loss_loop = compute_modified_subtb_core(
            log_probs,
            eos_logprobs,
            log_reward,
            mask,
            subtb_lambda=0.9,
            max_residual=50.0,
            _force_loop=True,
        )

        loss_vec = compute_modified_subtb_core(
            log_probs,
            eos_logprobs,
            log_reward,
            mask,
            subtb_lambda=0.9,
            max_residual=50.0,
            _force_loop=False,
        )

        torch.testing.assert_close(loss_loop, loss_vec, atol=1e-4, rtol=1e-4)

    def test_single_token(self):
        """T=1 case works correctly for both paths."""
        B = 2
        torch.manual_seed(42)
        log_probs = torch.randn(B, 1)
        eos_logprobs = torch.randn(B, 1)
        log_reward = torch.randn(B)
        mask = torch.ones(B, 1)

        loss_loop = compute_modified_subtb_core(
            log_probs,
            eos_logprobs,
            log_reward,
            mask,
            subtb_lambda=0.9,
            max_residual=50.0,
            _force_loop=True,
        )

        loss_vec = compute_modified_subtb_core(
            log_probs,
            eos_logprobs,
            log_reward,
            mask,
            subtb_lambda=0.9,
            max_residual=50.0,
            _force_loop=False,
        )

        torch.testing.assert_close(loss_loop, loss_vec, atol=1e-4, rtol=1e-4)

    def test_all_masked_batch(self):
        """Fully masked batch produces identical results (edge case)."""
        B, T = 2, 5
        torch.manual_seed(42)
        log_probs = torch.randn(B, T)
        eos_logprobs = torch.randn(B, T)
        log_reward = torch.randn(B)
        mask = torch.zeros(B, T)

        loss_loop = compute_modified_subtb_core(
            log_probs,
            eos_logprobs,
            log_reward,
            mask,
            subtb_lambda=0.9,
            max_residual=50.0,
            _force_loop=True,
        )

        loss_vec = compute_modified_subtb_core(
            log_probs,
            eos_logprobs,
            log_reward,
            mask,
            subtb_lambda=0.9,
            max_residual=50.0,
            _force_loop=False,
        )

        torch.testing.assert_close(loss_loop, loss_vec, atol=1e-4, rtol=1e-4)

    def test_clamping_equivalence(self):
        """Large residuals are clamped identically."""
        B, T = 2, 8
        torch.manual_seed(42)
        log_probs = torch.randn(B, T) * 100  # large values to trigger clamping
        eos_logprobs = torch.randn(B, T) * 100
        log_reward = torch.randn(B) * 100
        mask = torch.ones(B, T)

        loss_loop = compute_modified_subtb_core(
            log_probs,
            eos_logprobs,
            log_reward,
            mask,
            subtb_lambda=0.9,
            max_residual=50.0,
            _force_loop=True,
        )

        loss_vec = compute_modified_subtb_core(
            log_probs,
            eos_logprobs,
            log_reward,
            mask,
            subtb_lambda=0.9,
            max_residual=50.0,
            _force_loop=False,
        )

        torch.testing.assert_close(loss_loop, loss_vec, atol=1e-4, rtol=1e-4)
