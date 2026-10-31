"""Policy-owned trajectory normalizer for SkyRL's FSDP worker."""

from __future__ import annotations

from collections.abc import Mapping

import ray
import torch
from skyrl.backends.skyrl_train.weight_sync.sources import FsdpWeightSource
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase
from transformers import get_scheduler

_TB_DIAGNOSTIC_MEANS = (
    "loss_metrics/tb_loss_unweighted",
    "loss_metrics/mean_log_pf",
    "loss_metrics/mean_log_reward",
    "loss_metrics/lineage_loss_weight",
)


class LanguageModelWeights(FsdpWeightSource):
    """Use the native weight stream without sending objective state to vLLM."""

    def metadata(self):
        return [item for item in super().metadata() if item.name != f"{self.weight_prefix}log_z"]

    def __iter__(self):
        for name, tensor in super().__iter__():
            if name != f"{self.weight_prefix}log_z":
                yield name, tensor


class GFNPolicyWorker(FSDPPolicyWorkerBase):
    """Keep log Z in the policy's distributed optimizer and checkpoint."""

    _minibatch_valid_trajectories: int | None = None

    def _seq_parallel_monkey_patch(self, model):
        # SkyRL 0.4 invokes this construction hook before strategy.prepare().
        super()._seq_parallel_monkey_patch(model)
        device = next(model.parameters()).device
        model.register_parameter(
            "log_z",
            torch.nn.Parameter(
                torch.full((1,), self.cfg.algorithm.logZ_init, dtype=torch.float32, device=device)
            ),
        )
        model.register_forward_hook(self._capture_log_z)

    def _capture_log_z(self, model, args, output):
        # Read while FSDP has unsharded the parameter, before its post-forward hook.
        self._forward_log_z = model.log_z.float().sum()

    def _current_log_z(self):
        if not hasattr(self, "_forward_log_z"):
            raise RuntimeError("The policy must run a forward pass before the TB loss")
        return self._forward_log_z

    def init_model(self, model_path, num_training_steps=None):
        if self.cfg.policy.inference_only_init:
            raise ValueError("TB requires a trainable policy and normalizer")
        super().init_model(model_path, num_training_steps)
        normalizer = self.model.model.log_z
        for group in self.optimizer.param_groups:
            group["params"] = [param for param in group["params"] if param is not normalizer]
        self.optimizer.add_param_group(
            {
                "params": [normalizer],
                "lr": self.cfg.algorithm.logZ_lr,
                "initial_lr": self.cfg.algorithm.logZ_lr,
                "weight_decay": 0.0,
            }
        )
        optimizer_config = self.cfg.policy.optimizer_config
        self.scheduler = get_scheduler(
            optimizer_config.scheduler,
            self.optimizer,
            num_warmup_steps=optimizer_config.num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        self.cfg.algorithm.logZ_module = self._current_log_z

    def _resolve_global_valid_trajectories(self, data, local_valid: int) -> int:
        metadata = getattr(data, "metadata", None)
        if isinstance(metadata, Mapping) and metadata.get("global_valid_trajectories") is not None:
            return int(metadata["global_valid_trajectories"])
        if isinstance(data, Mapping) and data.get("global_valid_trajectories") is not None:
            raw_global = data["global_valid_trajectories"]
            if isinstance(raw_global, torch.Tensor) and raw_global.numel() > 0:
                return int(raw_global.flatten()[0].item())
            if isinstance(raw_global, int | float) and not isinstance(raw_global, bool):
                return int(raw_global)

        strategy = getattr(self, "strategy", None)
        if strategy is not None and hasattr(strategy, "all_reduce"):
            device_mesh = getattr(self, "device_mesh", None)
            dp_group = (
                device_mesh.get_group("dp")
                if device_mesh is not None and hasattr(device_mesh, "get_group")
                else None
            )
            reduced = strategy.all_reduce(
                {"valid_trajectories": float(local_valid)},
                op="sum",
                group=dp_group,
            )
            return int(round(float(reduced["valid_trajectories"])))

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            loss_mask = data.get("loss_mask") if hasattr(data, "get") else None
            device = (
                loss_mask.device if isinstance(loss_mask, torch.Tensor) else torch.device("cpu")
            )
            count_tensor = torch.tensor([local_valid], dtype=torch.long, device=device)
            torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.SUM)
            return int(count_tensor.item())

        dp_size = int(getattr(getattr(self, "mesh_rank", None), "dp_size", 1) or 1)
        if dp_size > 1:
            raise RuntimeError(
                "Global valid-trajectory count across DP ranks requires "
                "strategy.all_reduce, torch.distributed, or global_valid_trajectories metadata"
            )
        return local_valid

    def forward_backward(
        self,
        data,
        loss_fn=None,
        loss_fn_config=None,
        return_per_token_outputs: bool = True,
    ):
        loss_mask = data.get("loss_mask") if hasattr(data, "get") else None
        if isinstance(loss_mask, torch.Tensor):
            local_valid = int((loss_mask.float().sum(dim=-1) > 0).sum().item())
            self._minibatch_valid_trajectories = self._resolve_global_valid_trajectories(
                data,
                local_valid,
            )
        else:
            self._minibatch_valid_trajectories = None
        from skyrl.backends.skyrl_train.workers.worker_utils import BaseBatchIterator

        orig_batch_to_exp = BaseBatchIterator.batch_to_experience

        def _batch_to_exp_with_lineage(batch):
            exp = orig_batch_to_exp(batch)
            if hasattr(batch, "get"):
                lw = batch.get("lineage_loss_weights")
                if lw is not None:
                    exp.lineage_loss_weights = lw
            batch_meta = getattr(batch, "metadata", None)
            if isinstance(batch_meta, Mapping) and "lineage_loss_weight" in batch_meta:
                exp.lineage_loss_weight = batch_meta["lineage_loss_weight"]
            return exp

        BaseBatchIterator.batch_to_experience = staticmethod(_batch_to_exp_with_lineage)
        self._minibatch_diagnostics = dict.fromkeys(_TB_DIAGNOSTIC_MEANS, 0.0)
        self._minibatch_diagnostics["valid_trajectories"] = 0.0
        try:
            output = super().forward_backward(
                data,
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
                return_per_token_outputs=return_per_token_outputs,
            )
            if "loss_metrics/tb_loss_unweighted" in output.metrics:
                # Reduce once after all local microbatches, whose counts can differ by rank.
                totals = self.strategy.all_reduce(
                    self._minibatch_diagnostics,
                    op="sum",
                    group=self.device_mesh.get_group("dp"),
                )
                valid = totals["valid_trajectories"]
                for key in _TB_DIAGNOSTIC_MEANS:
                    output.metrics[key] = totals[key] / valid if valid > 0 else 0.0
                output.metrics["loss_metrics/valid_trajectories"] = valid
                if valid == 0:
                    output.metrics["loss_metrics/lineage_loss_weight_min"] = 0.0
                    output.metrics["loss_metrics/lineage_loss_weight_max"] = 0.0
            return output
        finally:
            BaseBatchIterator.batch_to_experience = staticmethod(orig_batch_to_exp)
            self._minibatch_valid_trajectories = None
            self._minibatch_diagnostics.clear()

    @staticmethod
    def _extract_experience_lineage(experience):
        mb_weights = getattr(experience, "lineage_loss_weights", None)
        if mb_weights is None and isinstance(experience, Mapping):
            mb_weights = experience.get("lineage_loss_weights")
        exp_meta = getattr(experience, "metadata", None)
        if mb_weights is None and isinstance(exp_meta, Mapping):
            mb_weights = exp_meta.get("lineage_loss_weights")

        mb_weight = getattr(experience, "lineage_loss_weight", None)
        if mb_weight is None and isinstance(experience, Mapping):
            mb_weight = experience.get("lineage_loss_weight")
        if mb_weight is None and isinstance(exp_meta, Mapping):
            mb_weight = exp_meta.get("lineage_loss_weight")
        return mb_weights, mb_weight

    def _forward_backward_micro(
        self,
        experience,
        microbatch_weight: float,
        loss_fn=None,
        loss_fn_config=None,
        return_per_token_outputs: bool = True,
    ):
        dp_size = int(getattr(getattr(self, "mesh_rank", None), "dp_size", 1) or 1)
        minibatch_valid = getattr(self, "_minibatch_valid_trajectories", None)
        mb_lineage_weights, mb_lineage_weight = self._extract_experience_lineage(experience)
        algorithm = self.cfg.algorithm
        prev_mb_weight = getattr(algorithm, "microbatch_weight", None)
        prev_dp_size = getattr(algorithm, "dp_size", 1)
        prev_mb_valid = getattr(algorithm, "minibatch_valid_trajectories", None)
        prev_lineage_weights = getattr(algorithm, "lineage_loss_weights", None)
        prev_lineage_weight = getattr(algorithm, "lineage_loss_weight", 1.0)
        algorithm.microbatch_weight = float(microbatch_weight)
        algorithm.dp_size = dp_size
        algorithm.minibatch_valid_trajectories = minibatch_valid
        if mb_lineage_weights is not None:
            algorithm.lineage_loss_weights = mb_lineage_weights
        if mb_lineage_weight is not None:
            algorithm.lineage_loss_weight = mb_lineage_weight
        try:
            metrics = super()._forward_backward_micro(
                experience,
                microbatch_weight,
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
                return_per_token_outputs=return_per_token_outputs,
            )
            if "loss_metrics/tb_loss_unweighted" in metrics:
                mask = experience.loss_mask
                valid = (
                    int((mask.float().sum(dim=-1) > 0).sum().item())
                    if mask is not None
                    else len(experience.sequences)
                )
                self._minibatch_diagnostics["valid_trajectories"] += valid
                for key in _TB_DIAGNOSTIC_MEANS:
                    self._minibatch_diagnostics[key] += metrics[key] * valid
            return metrics
        finally:
            algorithm.microbatch_weight = prev_mb_weight
            algorithm.dp_size = prev_dp_size
            algorithm.minibatch_valid_trajectories = prev_mb_valid
            algorithm.lineage_loss_weights = prev_lineage_weights
            algorithm.lineage_loss_weight = prev_lineage_weight

    def _build_weight_source(self, dtype, backend):
        if backend == "sharded_rdt":
            raise ValueError(
                "TB objective state filtering requires the native nccl or delta stream"
            )
        source = super()._build_weight_source(dtype, backend)
        return LanguageModelWeights(source.model, dtype, source.weight_prefix)


PolicyWorker = ray.remote(num_gpus=1)(GFNPolicyWorker)
