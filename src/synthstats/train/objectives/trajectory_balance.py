"""TB and SubTB losses for SkyRL integration.

TB:    L = (logZ + sum(log_pi) - log_R)^2
SubTB: L = Sigma_{len} lambda^{len-1} * (delta_cumsum[:, len:] - delta_cumsum[:, :-len])^2

tb_identity estimator passes log_rewards through as advantages.
logZ (and eos_logprobs for SubTB) injected via config by GFlowNetTrainer.

# Config interface (injected by the trainer before each step):
# - logZ: float (required for trajectory_balance and modified_subtb)
# - _logZ_tensor: Tensor (optional; preferred over logZ if present, preserves gradient)
# - _eos_logprobs: Tensor [B, T] (optional, for modified_subtb)
# - tb_max_residual: float (optional, default 100.0)
# - subtb_lambda: float (optional, default 0.9)
# - lineage_loss_weights: rollout-level weights (optional; preferred for TB)
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from synthstats.core.config_utils import config_get as _config_get
from synthstats.core.constants import SUBTB_LAMBDA_DEFAULT, TB_MAX_RESIDUAL_DEFAULT
from synthstats.core.math_utils import graph_connected_zero
from synthstats.train.objectives.losses import (
    compute_modified_subtb_core_terms,
    require_finite_log_rewards,
    vanilla_tb_residual,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def _lineage_loss_weight(config: DictConfig | dict[str, Any]) -> float:
    raw_weight = _config_get(config, "lineage_loss_weight", 1.0)
    if isinstance(raw_weight, bool):
        return 1.0
    if isinstance(raw_weight, int | float):
        weight = float(raw_weight)
        if math.isfinite(weight):
            return weight
    return 1.0


def _lineage_loss_weights(
    config: DictConfig | dict[str, Any],
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor | None:
    raw_weights = _config_get(config, "lineage_loss_weights", None)
    if raw_weights is None:
        return None

    if isinstance(raw_weights, Tensor):
        weights = raw_weights.to(dtype=dtype, device=device).flatten()
    elif isinstance(raw_weights, list | tuple):
        parsed: list[float] = []
        for value in raw_weights:
            if isinstance(value, bool):
                parsed.append(float(value))
            elif isinstance(value, int | float):
                candidate = float(value)
                if not math.isfinite(candidate):
                    return None
                parsed.append(candidate)
            else:
                return None
        weights = torch.tensor(parsed, dtype=dtype, device=device)
    else:
        return None

    if weights.numel() != batch_size:
        return None
    return weights


def compute_tb_identity_advantage(
    token_level_rewards: Tensor,
    response_mask: Tensor,
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """Pass log_rewards through as 'advantages' for TB training."""
    with torch.no_grad():
        trajectory_rewards = (token_level_rewards * response_mask).sum(dim=-1, keepdim=True)
        log_rewards = trajectory_rewards.expand_as(token_level_rewards) * response_mask

    return log_rewards, log_rewards  # (advantages, returns)


def compute_tb_log_reward_advantage(
    token_level_rewards: Tensor,
    response_mask: Tensor,
    *,
    config: Any,
    index: Any = None,
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """Transform raw rewards once to log(max(R, reward_floor)).

    TB targets a distribution proportional to max(R, reward_floor), so a
    zero reward counts as the declared floor. Masked observations do not
    contribute to a trajectory's reward.
    """
    floor = _config_get(config, "reward_floor", None)
    if (
        isinstance(floor, bool)
        or not isinstance(floor, int | float)
        or not math.isfinite(floor)
        or floor <= 0
    ):
        raise ValueError("TB requires a declared positive finite reward_floor")
    with torch.no_grad():
        rewards = torch.where(response_mask.bool(), token_level_rewards, 0).sum(-1, keepdim=True)
        invalid = (~torch.isfinite(rewards) | (rewards < 0)).flatten()
        if bool(invalid.any()):
            rows = invalid.nonzero().flatten().tolist()
            names = [str(index[row]) if index is not None else str(row) for row in rows]
            raise ValueError(f"Invalid nonnegative raw reward for trajectories: {', '.join(names)}")
        log_rewards = rewards.clamp_min(floor).log().expand_as(token_level_rewards)
        advantages = torch.where(response_mask.bool(), log_rewards, 0)
    return advantages, advantages


def _prepare_logz(
    config: DictConfig | dict[str, Any],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Extract logZ from config and cast to the given dtype/device.

    Raises RuntimeError if neither _logZ_tensor nor logZ is present.
    Using a tensor preserves the gradient for logZ learning; a plain float
    becomes a constant leaf.
    """
    logZ_tensor = getattr(config, "_logZ_tensor", None)
    logZ_val = _config_get(config, "logZ", None)

    if logZ_tensor is None and logZ_val is None:
        raise RuntimeError(
            "TB logZ not found in config. GFlowNetTrainer must set config.logZ "
            "before loss computation."
        )

    if logZ_tensor is not None:
        return logZ_tensor.to(dtype=dtype, device=device)
    return torch.tensor(logZ_val, dtype=dtype, device=device)


def compute_trajectory_balance_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    config: DictConfig | dict[str, Any],
    loss_mask: Tensor | None = None,
    rollout_logprobs: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """TB loss: L = (logZ + sum(log_pi) - log_R)^2. Returns (loss, clip_ratio=0.0)."""
    logZ = _prepare_logz(config, log_probs.dtype, log_probs.device)
    max_residual = _config_get(config, "tb_max_residual", TB_MAX_RESIDUAL_DEFAULT)

    log_rewards = advantages

    if loss_mask is not None:
        mask_f = loss_mask.float()
        masked_logprobs = (log_probs * mask_f).sum(dim=-1)  # [B]
        token_counts = mask_f.sum(dim=-1)  # [B]
        log_rewards_seq = torch.where(
            token_counts > 0,
            (log_rewards * mask_f).sum(dim=-1) / token_counts,
            torch.zeros_like(token_counts),
        )
        valid_traj = token_counts > 0  # [B]
    else:
        masked_logprobs = log_probs.sum(dim=-1)  # [B]
        log_rewards_seq = log_rewards.mean(dim=-1)  # [B]
        valid_traj = torch.ones(log_probs.shape[0], dtype=torch.bool, device=log_probs.device)

    safe_log_rewards = require_finite_log_rewards(log_rewards_seq)

    residual = vanilla_tb_residual(logZ, masked_logprobs, safe_log_rewards, max_residual)
    squared_residual = residual**2
    lineage_weights = _lineage_loss_weights(
        config,
        batch_size=masked_logprobs.shape[0],
        dtype=log_probs.dtype,
        device=log_probs.device,
    )
    if lineage_weights is not None and not bool(valid_traj.all()):
        lineage_weights = lineage_weights[valid_traj]

    # empty-mask trajectories would give wrong gradient
    if valid_traj.all():
        unweighted_loss = squared_residual.mean()
        if lineage_weights is not None and lineage_weights.numel() == squared_residual.numel():
            loss = (squared_residual * lineage_weights).mean()
            lineage_weight = lineage_weights.mean().item()
            lineage_weight_min = lineage_weights.min().item()
            lineage_weight_max = lineage_weights.max().item()
        else:
            lineage_weight = _lineage_loss_weight(config)
            loss = unweighted_loss * log_probs.new_tensor(lineage_weight)
            lineage_weight_min = lineage_weight
            lineage_weight_max = lineage_weight
    else:
        valid_count = valid_traj.sum()
        if valid_count > 0:
            valid_squared_residual = squared_residual[valid_traj]
            unweighted_loss = valid_squared_residual.sum() / valid_count
            if (
                lineage_weights is not None
                and lineage_weights.numel() == valid_squared_residual.numel()
            ):
                loss = (valid_squared_residual * lineage_weights).sum() / valid_count
                lineage_weight = lineage_weights.mean().item()
                lineage_weight_min = lineage_weights.min().item()
                lineage_weight_max = lineage_weights.max().item()
            else:
                lineage_weight = _lineage_loss_weight(config)
                loss = unweighted_loss * log_probs.new_tensor(lineage_weight)
                lineage_weight_min = lineage_weight
                lineage_weight_max = lineage_weight
        else:
            logger.warning("all-empty loss_mask in TB loss; batch will not learn")
            unweighted_loss = graph_connected_zero(masked_logprobs, logZ)
            lineage_weight = _lineage_loss_weight(config)
            loss = unweighted_loss * log_probs.new_tensor(lineage_weight)
            lineage_weight_min = lineage_weight
            lineage_weight_max = lineage_weight

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"TB loss: {loss.item():.4f}, logZ: {logZ.item():.4f}, "
            f"mean_log_pi: {masked_logprobs.mean().item():.4f}, "
            f"mean_log_R: {safe_log_rewards.mean().item():.4f}"
        )

    return loss, {
        "clip_ratio": 0.0,
        "lineage_loss_weight": lineage_weight,
        "lineage_loss_weight_min": lineage_weight_min,
        "lineage_loss_weight_max": lineage_weight_max,
    }  # SkyRL iterates .items() on this


def compute_modified_subtb_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    config: DictConfig | dict[str, Any],
    loss_mask: Tensor | None = None,
    rollout_logprobs: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Termination-potential SubTB under a shared unit early-stopping reward."""
    eos_logprobs = _config_get(config, "_eos_logprobs", None)
    if eos_logprobs is None:
        raise ValueError("SubTB requires aligned current-policy EOS log probabilities")

    subtb_lambda = _config_get(config, "subtb_lambda", SUBTB_LAMBDA_DEFAULT)
    max_residual = _config_get(config, "tb_max_residual", TB_MAX_RESIDUAL_DEFAULT)

    B, T = log_probs.shape
    device = log_probs.device
    dtype = log_probs.dtype

    # cast eos_logprobs to match device/dtype
    if isinstance(eos_logprobs, Tensor):
        eos_logprobs = eos_logprobs.to(dtype=dtype, device=device)
    else:
        eos_logprobs = torch.tensor(eos_logprobs, dtype=dtype, device=device)

    # handle mask
    if loss_mask is not None:
        mask = loss_mask.float()
    else:
        mask = torch.ones(B, T, device=device, dtype=dtype)

    # advantages is actually log_rewards (from tb_identity estimator)
    log_rewards = advantages

    eos_log_rewards = _config_get(config, "_eos_log_rewards", None)
    if eos_log_rewards is not None and not isinstance(eos_log_rewards, Tensor):
        eos_log_rewards = torch.tensor(eos_log_rewards, dtype=dtype, device=device)

    per_row_numerators, per_row_denominators = compute_modified_subtb_core_terms(
        log_probs=log_probs,
        eos_logprobs=eos_logprobs,
        log_rewards=log_rewards,
        mask=mask,
        subtb_lambda=subtb_lambda,
        max_residual=max_residual,
        terminal_is_eos=_config_get(config, "_terminal_is_eos", None),
        eos_log_rewards=eos_log_rewards,
    )
    unweighted_loss = per_row_numerators.sum() / per_row_denominators.sum().clamp(min=1.0)
    lineage_weights = _lineage_loss_weights(
        config,
        batch_size=B,
        dtype=log_probs.dtype,
        device=log_probs.device,
    )
    if lineage_weights is not None and lineage_weights.numel() == per_row_numerators.numel():
        weighted_numerator = (per_row_numerators * lineage_weights).sum()
        weighted_denominator = (per_row_denominators * lineage_weights).sum()
        loss = weighted_numerator / weighted_denominator.clamp(min=1.0)
        lineage_weight = lineage_weights.mean().item()
        lineage_weight_min = lineage_weights.min().item()
        lineage_weight_max = lineage_weights.max().item()
    else:
        lineage_weight = _lineage_loss_weight(config)
        loss = unweighted_loss * log_probs.new_tensor(lineage_weight)
        lineage_weight_min = lineage_weight
        lineage_weight_max = lineage_weight

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"SubTB loss: {loss.item():.4f}, lambda: {subtb_lambda}, seq_len: {T}, batch_size: {B}"
        )

    return loss, {
        "clip_ratio": 0.0,
        "lineage_loss_weight": lineage_weight,
        "lineage_loss_weight_min": lineage_weight_min,
        "lineage_loss_weight_max": lineage_weight_max,
    }  # SkyRL iterates .items() on this
