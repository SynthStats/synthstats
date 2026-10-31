"""TB backward must visit every sharded policy on a masked microbatch."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from synthstats.train.objectives.gfn_losses import gfn_tb_loss


class _ShardedPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 2)])
        self.register_parameter("log_z", nn.Parameter(torch.ones(1)))
        self.register_forward_hook(self._capture_log_z)

    def _capture_log_z(self, model, args, output):
        # Capture while the root parameter is unsharded, as the policy worker does.
        self.forward_log_z = model.log_z.float().sum()

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs).tanh()
        return inputs.log_softmax(dim=-1)


def _backward_masked_microbatch(rank: int, rendezvous: str, masked_rank: int):
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        torch.manual_seed(7)
        mesh = init_device_mesh("cpu", (2,))
        policy = _ShardedPolicy()
        for layer in policy.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(policy, mesh=mesh)
        cfg = SimpleNamespace(logZ_module=lambda: policy.forward_log_z, tb_max_residual=100.0)

        # A valid microbatch precedes the asymmetric one in the same training step.
        for microbatch in range(2):
            log_probs = policy(torch.ones(2, 4))
            mask = torch.ones_like(log_probs)
            if microbatch == 1 and rank == masked_rank:
                mask.zero_()
            loss, _ = gfn_tb_loss(
                log_probs, log_probs.detach(), torch.zeros_like(log_probs), cfg, loss_mask=mask
            )
            loss.backward()

        # Gathering completion after backward catches a rank that skips FSDP collectives.
        completed = [None, None]
        dist.all_gather_object(completed, rank)
        assert completed == [0, 1]
        for name, parameter in policy.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad.to_local()).all(), name
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required")
@pytest.mark.parametrize("masked_rank", [0, 1])
def test_tb_fsdp_policy_backward_completes_with_a_masked_microbatch(
    tmp_path: Path, masked_rank: int
):
    mp.spawn(
        _backward_masked_microbatch,
        args=((tmp_path / "gloo").as_uri(), masked_rank),
        nprocs=2,
        join=True,
    )
