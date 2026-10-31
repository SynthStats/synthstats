from __future__ import annotations

import pytest
import torch

from synthstats.train.objectives.gfn_losses import gfn_subtb_loss, gfn_tb_loss


class TestGFNLosses:
    def test_gfn_tb_loss_computes(self):
        from synthstats.integrations.skyrl.trainer import LogZModule

        batch_size, seq_len = 2, 5
        log_probs = torch.randn(batch_size, seq_len)
        advantages = torch.tensor([[1.5] * seq_len, [-0.3] * seq_len])

        class FakeConfig:
            logZ_module = LogZModule(0.0)
            logZ_lr = 0.025
            tb_max_residual = 100.0

        loss, metrics = gfn_tb_loss(
            log_probs,
            log_probs,
            advantages,
            FakeConfig(),
            loss_mask=torch.ones(batch_size, seq_len),
        )

        assert loss.shape == ()
        assert loss.requires_grad
        assert metrics["clip_ratio"] == pytest.approx(0.0)
        assert "tb_loss" in metrics
        assert "logZ" in metrics
        assert metrics["logZ_lr"] == pytest.approx(0.025)

    def test_gfn_tb_loss_applies_per_sample_lineage_weights(self):
        from synthstats.integrations.skyrl.trainer import LogZModule

        log_probs = torch.tensor([[-1.0, -1.0], [-2.0, -2.0]])
        advantages = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

        class WeightedConfig:
            logZ_module = LogZModule(0.0)
            tb_max_residual = 100.0
            lineage_loss_weights = [2.0, 1.0]

        weighted_loss, metrics = gfn_tb_loss(
            log_probs,
            log_probs,
            advantages,
            WeightedConfig(),
            loss_mask=torch.ones(2, 2),
        )

        expected = torch.tensor((2.0 * 9.0 + 1.0 * 25.0) / 2.0)
        assert torch.isclose(weighted_loss, expected)
        assert metrics["lineage_loss_weight"] == pytest.approx(1.5)

    def test_gfn_subtb_loss_rejects_missing_eos(self):
        from synthstats.integrations.skyrl.trainer import LogZModule

        class FakeConfig:
            logZ_module = LogZModule(0.0)
            tb_max_residual = 100.0
            _eos_logprobs = None

        with pytest.raises(ValueError, match="requires aligned current-policy EOS"):
            gfn_subtb_loss(
                torch.randn(2, 5),
                torch.randn(2, 5),
                torch.ones(2, 5),
                FakeConfig(),
                loss_mask=torch.ones(2, 5),
            )
