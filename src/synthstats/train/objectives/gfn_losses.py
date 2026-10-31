"""GFlowNet losses."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Protocol

import torch
from torch import Tensor

from synthstats.core.constants import (
    LOGZ_LR_DEFAULT,
    SUBTB_LAMBDA_DEFAULT,
    TB_MAX_RESIDUAL_DEFAULT,
)
from synthstats.core.math_utils import drop_empty_mask_rows, graph_connected_zero
from synthstats.train.objectives.losses import (
    compute_modified_subtb_core_terms,
    require_finite_log_rewards,
    vanilla_tb_residual,
)

if TYPE_CHECKING:
    import torch.nn as nn


class GFNLossConfig(Protocol):
    """Config interface for GFlowNet loss functions."""

    logZ_module: nn.Module
    tb_max_residual: float
    subtb_lambda: float
    _eos_logprobs: Tensor | None
    lineage_loss_weight: float
    lineage_loss_weights: Tensor | list[float] | None
    microbatch_weight: float | None
    dp_size: int
    minibatch_valid_trajectories: int | None


logger = logging.getLogger(__name__)


def compute_gfn_microbatch_scale(
    config: GFNLossConfig,
    *,
    microbatch_rows: int,
    valid_rows: int,
) -> float:
    """Prescale a microbatch mean loss for SkyRL's sum-across-microbatches/DP worker."""
    if valid_rows <= 0 or microbatch_rows <= 0:
        return 0.0

    dp_size_raw = getattr(config, "dp_size", 1)
    dp_size = (
        int(dp_size_raw)
        if not isinstance(dp_size_raw, bool)
        and isinstance(dp_size_raw, int | float)
        and int(dp_size_raw) > 0
        else 1
    )

    minibatch_valid_raw = getattr(config, "global_valid_trajectories", None)
    if minibatch_valid_raw is None:
        minibatch_valid_raw = getattr(config, "minibatch_valid_trajectories", None)
    if not isinstance(minibatch_valid_raw, bool) and isinstance(minibatch_valid_raw, int | float):
        global_valid = int(minibatch_valid_raw)
        if global_valid <= 0:
            return 0.0
        return float(valid_rows) / float(global_valid)

    microbatch_weight_raw = getattr(config, "microbatch_weight", None)
    if (
        not isinstance(microbatch_weight_raw, bool)
        and isinstance(microbatch_weight_raw, int | float)
        and math.isfinite(float(microbatch_weight_raw))
        and float(microbatch_weight_raw) >= 0.0
    ):
        microbatch_share = float(microbatch_weight_raw) * (
            float(valid_rows) / float(microbatch_rows)
        )
        return microbatch_share / float(dp_size)

    return 1.0 / float(dp_size)


def _lineage_loss_weight(config: GFNLossConfig) -> float:
    raw_weight = getattr(config, "lineage_loss_weight", 1.0)
    if isinstance(raw_weight, bool):
        return 1.0
    if isinstance(raw_weight, int | float):
        weight = float(raw_weight)
        if math.isfinite(weight):
            return weight
    return 1.0


def _lineage_loss_weights(
    config: GFNLossConfig,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor | None:
    raw_weights = getattr(config, "lineage_loss_weights", None)
    if raw_weights is None:
        return None

    if isinstance(raw_weights, Tensor):
        weights = raw_weights.to(dtype=dtype, device=device).flatten()
    elif isinstance(raw_weights, list | tuple):
        parsed: list[float] = []
        for value in raw_weights:
            if isinstance(value, bool):
                parsed.append(float(value))
            elif isinstance(value, int | float) and math.isfinite(float(value)):
                parsed.append(float(value))
            else:
                return None
        weights = torch.tensor(parsed, dtype=dtype, device=device)
    else:
        return None

    if weights.numel() != batch_size:
        return None
    return weights


def _extract_log_rewards(
    advantages: Tensor,
    mask_f: Tensor,
    max_residual: float,
) -> Tensor:
    """Extract per-trajectory log rewards from broadcasted advantages."""
    token_counts = mask_f.sum(dim=-1).clamp(min=1.0)
    log_rewards = (advantages * mask_f).sum(dim=-1) / token_counts

    if torch.is_grad_enabled():
        with torch.no_grad():
            adv_mean = log_rewards.unsqueeze(-1) * mask_f
            adv_diff = ((advantages * mask_f) - adv_mean).abs().max()
            if adv_diff > 1e-4:
                logger.warning(
                    "advantages not constant across valid positions (max diff=%.6f); "
                    "SkyRL may have normalized them",
                    adv_diff.item(),
                )

    return require_finite_log_rewards(log_rewards)


def gfn_tb_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    config: GFNLossConfig,
    loss_mask: Tensor | None = None,
    rollout_logprobs: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """TB loss for the GFlowNet policy-loss interface."""
    del old_log_probs, rollout_logprobs

    logZ_module: nn.Module | None = getattr(config, "logZ_module", None)
    if logZ_module is None:
        raise RuntimeError(
            "config.logZ_module not set. GFlowNetTrainer must inject LogZModule "
            "before loss computation."
        )

    logZ: Tensor = logZ_module()
    max_residual = float(getattr(config, "tb_max_residual", TB_MAX_RESIDUAL_DEFAULT))

    if loss_mask is None:
        loss_mask = torch.ones_like(log_probs)
    mask_f = loss_mask.float()
    raw_batch_size = int(mask_f.shape[0])
    row_valid = mask_f.sum(dim=-1) > 0
    lineage_weights = _lineage_loss_weights(
        config,
        batch_size=raw_batch_size,
        dtype=log_probs.dtype,
        device=log_probs.device,
    )
    if lineage_weights is not None and not bool(row_valid.all()):
        lineage_weights = lineage_weights[row_valid]

    (log_probs, advantages, mask_f), _ = drop_empty_mask_rows(
        log_probs,
        advantages,
        mask_f,
        mask=mask_f,
    )

    if log_probs.numel() == 0:
        return graph_connected_zero(log_probs, logZ), {
            "clip_ratio": 0.0,
            "tb_loss": 0.0,
            "tb_loss_unweighted": 0.0,
            "microbatch_loss_scale": 0.0,
            "logZ": logZ.item(),
            "logZ_lr": float(getattr(config, "logZ_lr", LOGZ_LR_DEFAULT)),
            "mean_log_pf": 0.0,
            "mean_log_reward": 0.0,
            "lineage_loss_weight": 0.0,
            "lineage_loss_weight_min": math.inf,
            "lineage_loss_weight_max": -math.inf,
        }

    log_pf = (log_probs.float() * mask_f).sum(dim=-1)
    safe_log_rewards = _extract_log_rewards(advantages, mask_f, max_residual)

    residual = vanilla_tb_residual(logZ, log_pf, safe_log_rewards, max_residual)
    squared_residual = residual**2
    unweighted_loss = squared_residual.mean()
    if lineage_weights is not None and lineage_weights.numel() == squared_residual.numel():
        raw_loss = (squared_residual * lineage_weights).mean()
        lineage_weight = lineage_weights.mean().item()
        lineage_weight_min = lineage_weights.min().item()
        lineage_weight_max = lineage_weights.max().item()
    else:
        lineage_weight = _lineage_loss_weight(config)
        raw_loss = unweighted_loss * log_probs.new_tensor(lineage_weight)
        lineage_weight_min = lineage_weight
        lineage_weight_max = lineage_weight

    scale = compute_gfn_microbatch_scale(
        config,
        microbatch_rows=raw_batch_size,
        valid_rows=int(squared_residual.numel()),
    )
    loss = raw_loss * log_probs.new_tensor(scale)

    metrics = {
        "clip_ratio": 0.0,
        "tb_loss": loss.item(),
        "tb_loss_unweighted": unweighted_loss.item(),
        "microbatch_loss_scale": scale,
        "logZ": logZ.item(),
        "logZ_lr": float(getattr(config, "logZ_lr", LOGZ_LR_DEFAULT)),
        "mean_log_pf": log_pf.mean().item(),
        "mean_log_reward": safe_log_rewards.mean().item(),
        "lineage_loss_weight": lineage_weight,
        "lineage_loss_weight_min": lineage_weight_min,
        "lineage_loss_weight_max": lineage_weight_max,
    }
    return loss, metrics


def gfn_subtb_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    config: GFNLossConfig,
    loss_mask: Tensor | None = None,
    rollout_logprobs: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Modified SubTB loss for the GFlowNet policy-loss interface."""
    eos_logprobs: Tensor | None = getattr(config, "_eos_logprobs", None)
    if eos_logprobs is None:
        raise ValueError(
            "SubTB requires aligned current-policy EOS log probabilities; "
            "missing EOS data cannot be replaced by a TB objective."
        )

    del old_log_probs, rollout_logprobs

    subtb_lambda = float(getattr(config, "subtb_lambda", SUBTB_LAMBDA_DEFAULT))
    max_residual = float(getattr(config, "tb_max_residual", TB_MAX_RESIDUAL_DEFAULT))

    if loss_mask is None:
        loss_mask = torch.ones_like(log_probs)
    mask_f = loss_mask.float()
    raw_batch_size = int(mask_f.shape[0])
    row_valid = mask_f.sum(dim=-1) > 0
    lineage_weights = _lineage_loss_weights(
        config,
        batch_size=raw_batch_size,
        dtype=log_probs.dtype,
        device=log_probs.device,
    )
    if lineage_weights is not None and not bool(row_valid.all()):
        lineage_weights = lineage_weights[row_valid]

    if isinstance(eos_logprobs, Tensor) and eos_logprobs.dim() == 2:
        (log_probs, advantages, mask_f, eos_logprobs), _ = drop_empty_mask_rows(
            log_probs,
            advantages,
            mask_f,
            eos_logprobs,
            mask=mask_f,
        )
    else:
        (log_probs, advantages, mask_f), _ = drop_empty_mask_rows(
            log_probs,
            advantages,
            mask_f,
            mask=mask_f,
        )

    if log_probs.numel() == 0:
        logZ_module: nn.Module | None = getattr(config, "logZ_module", None)
        empty_metrics: dict[str, float] = {
            "clip_ratio": 0.0,
            "subtb_loss": 0.0,
            "subtb_loss_unweighted": 0.0,
            "microbatch_loss_scale": 0.0,
            "lineage_loss_weight": _lineage_loss_weight(config),
        }
        if logZ_module is not None:
            return logZ_module() * 0.0, empty_metrics
        return torch.tensor(0.0, device=log_probs.device, requires_grad=True), empty_metrics

    batch_size, seq_len = log_probs.shape

    if isinstance(eos_logprobs, Tensor):
        eos_logprobs = eos_logprobs.to(dtype=log_probs.dtype, device=log_probs.device)
    else:
        eos_logprobs = torch.tensor(
            eos_logprobs,
            dtype=log_probs.dtype,
            device=log_probs.device,
        )

    log_rewards = _extract_log_rewards(advantages, mask_f, max_residual)

    terminal_is_eos = getattr(config, "_terminal_is_eos", None)
    if terminal_is_eos is not None and not bool(row_valid.all()):
        terminal_is_eos = terminal_is_eos[row_valid]

    eos_log_rewards = getattr(config, "_eos_log_rewards", None)
    if eos_log_rewards is not None:
        if not isinstance(eos_log_rewards, Tensor):
            eos_log_rewards = torch.tensor(
                eos_log_rewards,
                dtype=log_probs.dtype,
                device=log_probs.device,
            )
        if not bool(row_valid.all()):
            eos_log_rewards = eos_log_rewards[row_valid]

    per_row_numerators, per_row_denominators = compute_modified_subtb_core_terms(
        log_probs=log_probs,
        eos_logprobs=eos_logprobs,
        log_rewards=log_rewards,
        mask=mask_f,
        subtb_lambda=subtb_lambda,
        max_residual=max_residual,
        terminal_is_eos=terminal_is_eos,
        eos_log_rewards=eos_log_rewards,
    )
    unweighted_loss = per_row_numerators.sum() / per_row_denominators.sum().clamp(min=1.0)
    if lineage_weights is not None and lineage_weights.numel() == per_row_numerators.numel():
        weighted_numerator = (per_row_numerators * lineage_weights).sum()
        weighted_denominator = (per_row_denominators * lineage_weights).sum()
        raw_loss = weighted_numerator / weighted_denominator.clamp(min=1.0)
        lineage_weight = lineage_weights.mean().item()
        lineage_weight_min = lineage_weights.min().item()
        lineage_weight_max = lineage_weights.max().item()
    else:
        lineage_weight = _lineage_loss_weight(config)
        raw_loss = unweighted_loss * log_probs.new_tensor(lineage_weight)
        lineage_weight_min = lineage_weight
        lineage_weight_max = lineage_weight

    scale = compute_gfn_microbatch_scale(
        config,
        microbatch_rows=raw_batch_size,
        valid_rows=batch_size,
    )
    loss = raw_loss * log_probs.new_tensor(scale)

    metrics = {
        "clip_ratio": 0.0,
        "subtb_loss": loss.item(),
        "subtb_loss_unweighted": unweighted_loss.item(),
        "microbatch_loss_scale": scale,
        "subtb_lambda": subtb_lambda,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "lineage_loss_weight": lineage_weight,
        "lineage_loss_weight_min": lineage_weight_min,
        "lineage_loss_weight_max": lineage_weight_max,
    }
    return loss, metrics


__all__ = [
    "GFNLossConfig",
    "compute_gfn_microbatch_scale",
    "gfn_subtb_loss",
    "gfn_tb_loss",
]
