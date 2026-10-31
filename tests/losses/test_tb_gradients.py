"""Trajectory-balance gradients at full sequence scale."""

from types import SimpleNamespace

import pytest
import torch

from synthstats.train.gfn_core import LogZModule
from synthstats.train.objectives.gfn_losses import gfn_tb_loss
from synthstats.train.objectives.losses import tb_loss


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_long_trajectory_retains_tb_gradient(registered, dtype):
    log_probs = torch.full((1, 200), -1.0, dtype=dtype, requires_grad=True)
    logz = LogZModule()
    mask = torch.ones_like(log_probs, dtype=torch.bool)
    if registered:
        loss, _ = gfn_tb_loss(
            log_probs,
            log_probs.detach(),
            torch.zeros_like(log_probs),
            SimpleNamespace(logZ_module=logz),
            mask,
        )
    else:
        loss = tb_loss(log_probs, mask, torch.zeros(1), logz())
    loss.backward()
    # Residual = -200; L = 40000; dL/dlogZ = dL/dlogpi_t = -400.
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(40000.0)
    torch.testing.assert_close(log_probs.grad.float(), torch.full((1, 200), -400.0))
    assert logz.logZ.grad.item() == pytest.approx(-400.0)
