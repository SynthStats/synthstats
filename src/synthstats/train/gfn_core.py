"""GFlowNet trainer core."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from synthstats.core.constants import LOGZ_LR_DEFAULT
from synthstats.train.lineage import (
    apply_lineage_batching,
    compute_lineage_loss_weight,
    compute_lineage_loss_weights,
    extract_numeric_rollout_metrics,
    extract_rollout_metric_sequences,
)

logger = logging.getLogger(__name__)


class LogZModule(nn.Module):
    """Learnable log partition function for GFlowNets."""

    def __init__(self, init_value: float = 0.0) -> None:
        super().__init__()
        self.logZ = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return self.logZ


class GFlowNetTrainerCore:
    """Shared GFlowNet trainer logic, independent of SkyRL."""

    logZ_module: LogZModule
    logZ_optimizer: torch.optim.Optimizer
    _gfn_step_count: int
    _pending_rollout_metrics: dict[str, Any] | None = None

    def init_gfn(
        self,
        logZ_init: float = 0.0,
        logZ_lr: float = LOGZ_LR_DEFAULT,
    ) -> None:
        self.logZ_module = LogZModule(logZ_init)
        self.logZ_optimizer = torch.optim.Adam(
            self.logZ_module.parameters(),
            lr=logZ_lr,
        )
        self._gfn_step_count = 0

    @property
    def logZ(self) -> torch.Tensor:
        return self.logZ_module.logZ

    def inject_logZ_into_config(self, config: Any) -> None:
        try:
            object.__setattr__(config, "logZ_module", self.logZ_module)
        except (AttributeError, TypeError):
            logger.warning("Could not inject logZ_module via setattr; loss will fail")

        logZ_val = self.logZ.item()
        if isinstance(config, dict):
            if "algorithm" in config:
                config["algorithm"]["logZ"] = logZ_val
            else:
                config["logZ"] = logZ_val
            return

        if hasattr(config, "algorithm"):
            config.algorithm.logZ = logZ_val
            return

        try:
            config.logZ = logZ_val
        except (AttributeError, TypeError):
            logger.debug("Could not set config.logZ directly")

    def merge_rollout_metrics(self, data: Any) -> dict[str, float]:
        """Normalize rollout metrics from the current batch into trainer metric space."""
        rollout_metrics = extract_numeric_rollout_metrics(data)
        if not rollout_metrics:
            return {}
        return {f"rollout_{key}": value for key, value in rollout_metrics.items()}

    def inject_lineage_loss_weight_into_config(
        self,
        config: Any,
        data: Any,
    ) -> dict[str, float]:
        """Inject the current batch's lineage-derived loss multiplier into config."""
        weights = compute_lineage_loss_weights(data, config)
        weight = compute_lineage_loss_weight(data, config)

        if isinstance(config, dict):
            algorithm = config.get("algorithm")
            if isinstance(algorithm, dict):
                algorithm["lineage_loss_weight"] = weight
                algorithm["lineage_loss_weights"] = weights
            else:
                config["lineage_loss_weight"] = weight
                config["lineage_loss_weights"] = weights
            return _lineage_weight_metrics(weight, weights)

        if hasattr(config, "trainer") and hasattr(config.trainer, "algorithm"):
            try:
                config.trainer.algorithm.lineage_loss_weight = weight
                config.trainer.algorithm.lineage_loss_weights = weights
                return _lineage_weight_metrics(weight, weights)
            except (AttributeError, TypeError):
                logger.debug("Could not inject lineage_loss_weight into config.trainer.algorithm")

        if hasattr(config, "algorithm"):
            try:
                config.algorithm.lineage_loss_weight = weight
                config.algorithm.lineage_loss_weights = weights
                return _lineage_weight_metrics(weight, weights)
            except (AttributeError, TypeError):
                logger.debug("Could not inject lineage_loss_weight into config.algorithm")

        try:
            config.lineage_loss_weight = weight
            config.lineage_loss_weights = weights
        except (AttributeError, TypeError):
            logger.debug("Could not set config.lineage_loss_weight directly")

        return _lineage_weight_metrics(weight, weights)

    def gfn_optimizer_step(self) -> dict[str, float]:
        self.logZ_optimizer.step()
        self.logZ_optimizer.zero_grad()
        self._gfn_step_count += 1
        return {"logZ": self.logZ.item(), "gfn_step": self._gfn_step_count}

    def gfn_state_dict(self) -> dict[str, Any]:
        return {
            "logZ": self.logZ.item(),
            "logZ_optimizer": self.logZ_optimizer.state_dict(),
            "gfn_step_count": self._gfn_step_count,
        }

    def gfn_load_state_dict(self, state: dict[str, Any]) -> None:
        with torch.no_grad():
            self.logZ_module.logZ.fill_(state["logZ"])
        if "logZ_optimizer" in state:
            self.logZ_optimizer.load_state_dict(state["logZ_optimizer"])
        self._gfn_step_count = state.get("gfn_step_count", 0)

    def to_checkpoint_state(self, step_count: int | None = None) -> Any:
        """Convert GFlowNet state into a unified CheckpointState."""
        from synthstats.train.checkpointing.base import CheckpointState

        return CheckpointState.from_gfn_state(
            self.gfn_state_dict(),
            step_count=step_count or self._gfn_step_count,
        )

    def load_from_checkpoint_state(self, state: Any) -> None:
        """Restore GFlowNet state from a unified CheckpointState."""
        gfn_state = state.to_gfn_state() if hasattr(state, "to_gfn_state") else state
        self.gfn_load_state_dict(gfn_state)

    @torch.no_grad()
    def _normalize_advantages(
        self,
        data: Any,
        mini_batch_boundaries: list[tuple[int, int]] | None = None,
        prompt_boundaries: list[tuple[int, int]] | None = None,
    ) -> Any:
        """Preserve raw trajectory log-rewards and apply rollout lineage controls."""
        del mini_batch_boundaries, prompt_boundaries
        return self.apply_batch_lineage_controls(data)

    def _ensure_batch_rollout_metrics(self, data: Any) -> None:
        if extract_rollout_metric_sequences(data) or extract_numeric_rollout_metrics(data):
            return
        metadata = getattr(data, "metadata", None)
        if not isinstance(metadata, dict):
            return
        pending = getattr(self, "_pending_rollout_metrics", None)
        if isinstance(pending, Mapping) and pending:
            metadata["rollout_metrics"] = deepcopy(dict(pending))
            self._pending_rollout_metrics = None
            return
        all_metrics = getattr(self, "all_metrics", None)
        if isinstance(all_metrics, Mapping) and (
            extract_rollout_metric_sequences({"rollout_metrics": all_metrics})
            or extract_numeric_rollout_metrics({"rollout_metrics": all_metrics})
        ):
            metadata["rollout_metrics"] = deepcopy(dict(all_metrics))

    def apply_batch_lineage_controls(self, data: Any) -> Any:
        """Apply configured rollout lineage batching and loss weighting on a batch."""
        self._ensure_batch_rollout_metrics(data)
        cfg = getattr(self, "cfg", None)
        batch_metrics = apply_lineage_batching(data, cfg) if cfg is not None else {}
        rollout_metrics = self.merge_rollout_metrics(data)
        lineage_metrics = (
            self.inject_lineage_loss_weight_into_config(cfg, data) if cfg is not None else {}
        )
        metadata = getattr(data, "metadata", None)
        weights = compute_lineage_loss_weights(data, cfg) if cfg is not None else None
        weight = lineage_metrics.get("lineage_loss_weight", 1.0)
        if isinstance(metadata, dict) and cfg is not None:
            metadata["lineage_loss_weight"] = weight
            metadata["lineage_loss_weights"] = weights
        if weights is not None and isinstance(data, dict):
            ref_tensor = next(
                (v for v in data.values() if isinstance(v, torch.Tensor) and v.dim() > 0),
                None,
            )
            if ref_tensor is not None:
                tensor_rows = int(ref_tensor.shape[0])
                pad_size = (
                    int(metadata.get("pad_size", 0) or 0) if isinstance(metadata, Mapping) else 0
                )
                if len(weights) == tensor_rows:
                    padded_weights: list[float] | None = list(weights)
                elif pad_size > 0 and len(weights) + pad_size == tensor_rows:
                    padded_weights = list(weights) + [1.0] * pad_size
                else:
                    padded_weights = None
                if padded_weights is not None:
                    data["lineage_loss_weights"] = torch.tensor(
                        padded_weights,
                        dtype=torch.float32,
                        device=ref_tensor.device,
                    )
        all_metrics = getattr(self, "all_metrics", None)
        if isinstance(all_metrics, dict):
            all_metrics.update(rollout_metrics)
            all_metrics.update(batch_metrics)
            all_metrics.update(lineage_metrics)
        return data


def resolve_logz_params(config: Any) -> tuple[float, float]:
    """Extract logZ init and learning-rate settings from dict or object config."""
    logZ_init = float(getattr(config, "logZ_init", 0.0))
    logZ_lr = float(getattr(config, "logZ_lr", LOGZ_LR_DEFAULT))
    if hasattr(config, "get"):
        logZ_init = float(config.get("logZ_init", logZ_init))
        logZ_lr = float(config.get("logZ_lr", logZ_lr))
    return logZ_init, logZ_lr


def _lineage_weight_metrics(
    mean_weight: float,
    weights: list[float] | None,
) -> dict[str, float]:
    metrics = {"lineage_loss_weight": mean_weight}
    if weights:
        metrics["lineage_loss_weight_min"] = min(weights)
        metrics["lineage_loss_weight_max"] = max(weights)
    return metrics


__all__ = [
    "GFlowNetTrainerCore",
    "LogZModule",
    "resolve_logz_params",
]
