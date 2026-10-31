import math
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from synthstats.train.gfn_core import LogZModule
from synthstats.train.objectives.gfn_losses import gfn_tb_loss


class _GlooStrategy:
    """Use real CPU collectives at the native worker's distributed boundary."""

    def all_reduce(self, data, op="mean", group=None):
        if isinstance(data, dict):
            return {key: self.all_reduce(value, op, group) for key, value in data.items()}
        value = torch.tensor(data, dtype=torch.float64)
        if op == "mean":
            value /= dist.get_world_size(group)
        reduction = {
            "mean": dist.ReduceOp.SUM,
            "sum": dist.ReduceOp.SUM,
            "min": dist.ReduceOp.MIN,
            "max": dist.ReduceOp.MAX,
        }[op]
        dist.all_reduce(value, op=reduction, group=group)
        return value.item()


def _run_worker(rank: int, rendezvous: str, case: str):
    from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch

    from synthstats.integrations.skyrl.policy_worker import FSDPPolicyWorkerBase, GFNPolicyWorker

    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        if case == "uneven":
            rows = [0, 3, 3, 3] if rank == 0 else [1, 2, 3, 3]
            log_probs = torch.tensor([[-1.0], [-3.0], [-5.0], [-9.0]])[rows]
            rewards = torch.tensor([[1.0], [2.0], [3.0], [9.0]])[rows]
            weights = torch.tensor([2.0, 4.0, 6.0, 99.0])[rows]
            mask = torch.tensor([[1.0], [1.0], [1.0], [0.0]])[rows]
            expected = (22.0, -3.0, 2.0, 4.0, 2.0, 6.0, 120.0)
        else:
            log_probs = torch.tensor([[-1.0], [-3.0]])
            rewards = torch.ones_like(log_probs)
            weights = torch.tensor([2.0, 4.0])
            populated = case != "empty" and rank != int(case)
            mask = torch.full_like(log_probs, int(populated))
            expected = (5.0, -2.0, 1.0, 3.0, 2.0, 4.0, 19.0)

        worker = object.__new__(GFNPolicyWorker)
        worker.strategy = _GlooStrategy()
        worker.device_mesh = SimpleNamespace(get_group=lambda name: dist.group.WORLD)
        worker.mesh_rank = SimpleNamespace(dp_size=2)
        worker.cfg = SimpleNamespace(
            micro_train_batch_size_per_gpu=2 if rank == 0 else 1,
            max_tokens_per_microbatch=0,
            algorithm=SimpleNamespace(
                logZ_module=LogZModule(1.0), logZ_lr=0.125, tb_max_residual=100.0
            ),
        )
        batch = TrainingInputBatch(
            {
                "sequences": torch.ones((len(mask), 2), dtype=torch.long),
                "action_log_probs": log_probs,
                "advantages": rewards,
                "loss_mask": mask,
                "lineage_loss_weights": weights,
            },
        )
        batch.metadata = {"response_length": 1}

        def loss_on_cpu(self, experience, microbatch_weight, **kwargs):
            loss, metrics = gfn_tb_loss(
                experience.action_log_probs,
                experience.action_log_probs,
                experience.advantages,
                self.cfg.algorithm,
                loss_mask=experience.loss_mask,
            )
            (loss * self.mesh_rank.dp_size).backward()
            return {f"loss_metrics/{key}": value for key, value in metrics.items()}

        original_micro = FSDPPolicyWorkerBase._forward_backward_micro
        FSDPPolicyWorkerBase._forward_backward_micro = loss_on_cpu
        try:
            output = worker.forward_backward(batch)
            # A second minibatch must not reuse the preceding diagnostic totals.
            batch["loss_mask"].zero_()
            empty_output = worker.forward_backward(batch)
        finally:
            FSDPPolicyWorkerBase._forward_backward_micro = original_micro

        keys = (
            "tb_loss_unweighted",
            "mean_log_pf",
            "mean_log_reward",
            "lineage_loss_weight",
            "lineage_loss_weight_min",
            "lineage_loss_weight_max",
            "tb_loss",
        )
        if case == "empty":
            expected = (0.0,) * len(keys)
        for key, value in zip(keys, expected, strict=True):
            assert output.metrics[f"loss_metrics/{key}"] == pytest.approx(value), key
            assert empty_output.metrics[f"loss_metrics/{key}"] == 0.0, key
        assert output.metrics["loss_metrics/logZ"] == pytest.approx(1.0)
        assert output.metrics["loss_metrics/logZ_lr"] == pytest.approx(0.125)
        assert all(math.isfinite(value) for value in empty_output.metrics.values())
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required")
@pytest.mark.parametrize("case", ["0", "1", "uneven", "empty"])
def test_worker_diagnostics_average_valid_trajectories(tmp_path: Path, case: str):
    pytest.importorskip("skyrl")
    mp.spawn(_run_worker, args=((tmp_path / "gloo").as_uri(), case), nprocs=2, join=True)
