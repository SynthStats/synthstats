from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from synthstats.train.gfn_core import LogZModule
from synthstats.train.objectives.gfn_losses import gfn_tb_loss


def test_tb_masked_rows_keep_metric_keys_and_order():
    cfg = SimpleNamespace(logZ_module=LogZModule(), tb_max_residual=100.0)
    lp = torch.tensor([[-1.0]], requires_grad=True)
    keys = []
    for mask in (torch.ones_like(lp), torch.zeros_like(lp)):
        _, metrics = gfn_tb_loss(lp, lp.detach(), torch.zeros_like(lp), cfg, loss_mask=mask)
        keys.append(tuple(metrics))
    assert keys[0] == keys[1]


class _GlooMetricStrategy:
    """Run the native metric reducer's collectives on CPU."""

    def all_reduce(self, data, op="mean", group=None):
        if isinstance(data, dict):
            return {key: self.all_reduce(value, op, group) for key, value in data.items()}
        scalar = torch.tensor(data, dtype=torch.float64)
        if op == "mean":
            scalar /= dist.get_world_size(group)
        reduction = {
            "mean": dist.ReduceOp.SUM,
            "sum": dist.ReduceOp.SUM,
            "min": dist.ReduceOp.MIN,
            "max": dist.ReduceOp.MAX,
        }[op]
        dist.all_reduce(scalar, op=reduction, group=group)
        return scalar.item()


def _reduce_masked_rank(rank: int, rendezvous: str, masked_rank: int):
    from skyrl.backends.skyrl_train.workers.worker_utils import all_reduce_metrics

    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        lp = torch.tensor([[-1.0], [-3.0]], requires_grad=True)
        cfg = SimpleNamespace(
            logZ_module=LogZModule(1.0),
            logZ_lr=0.125,
            tb_max_residual=100.0,
            dp_size=2,
            global_valid_trajectories=2,
            lineage_loss_weights=[2.0, 4.0],
        )
        loss, metrics = gfn_tb_loss(
            lp,
            lp.detach(),
            torch.ones_like(lp),
            cfg,
            loss_mask=torch.full_like(lp, int(rank != masked_rank)),
        )
        keys = [None, None]
        dist.all_gather_object(keys, tuple(metrics))
        assert keys[0] == keys[1]
        reduced = all_reduce_metrics(metrics, _GlooMetricStrategy(), sum_loss_metrics=True)
        # (2 * (-1)^2 + 4 * (-3)^2) / 2 = 19; the masked rank contributes zero.
        assert reduced["tb_loss"] == pytest.approx(19.0)
        assert reduced["lineage_loss_weight_min"] == pytest.approx(2.0)
        assert reduced["lineage_loss_weight_max"] == pytest.approx(4.0)
        assert reduced["logZ"] == pytest.approx(1.0)
        assert reduced["logZ_lr"] == pytest.approx(0.125)
        loss.backward()
        assert cfg.logZ_module.logZ.grad is not None
        assert cfg.logZ_module.logZ.grad.item() == pytest.approx(
            0.0 if rank == masked_rank else -14.0
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required")
@pytest.mark.parametrize("masked_rank", [0, 1])
def test_tb_metric_reduction_with_an_all_masked_rank(tmp_path: Path, masked_rank: int):
    pytest.importorskip("skyrl")
    mp.spawn(
        _reduce_masked_rank,
        args=((tmp_path / "gloo").as_uri(), masked_rank),
        nprocs=2,
        join=True,
    )
