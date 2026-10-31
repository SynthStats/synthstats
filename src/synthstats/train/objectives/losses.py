"""TB and Modified SubTB loss functions.

Canonical implementations shared by:
- train/objectives/trajectory_balance.py (SkyRL integration)

Vanilla TB: L = (logZ + sum(log_pi) - log_R)^2
Modified SubTB: Lambda-weighted sub-trajectory flow matching with EOS potentials

Research loss variants (TB+KL, FlowRL, VarGrad):
- compute_tb_loss_with_kl: TB with optional KL regularization
- compute_flowrl_loss: FlowRL (arXiv:2509.15207) with importance weights
- compute_vargrad_tb_loss: VarGrad TB (batch-estimated logZ, no learned parameter)
- estimate_log_partition_vargrad: Importance-sampling logZ estimator
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor

from synthstats.core.constants import SUBTB_LAMBDA_DEFAULT, TB_MAX_RESIDUAL_DEFAULT
from synthstats.core.math_utils import graph_connected_zero, sanitize_finite

logger = logging.getLogger(__name__)


def require_finite_log_rewards(log_rewards: Tensor) -> Tensor:
    """Reject invalid log rewards at the loss boundary."""
    if not bool(torch.isfinite(log_rewards).all()):
        raise ValueError("TB requires finite log rewards")
    return log_rewards


def _log_mean_exp(scores: Tensor, dim: int = 0) -> Tensor:
    """Numerically stable log(mean(exp(scores))) along *dim*."""
    K = scores.shape[dim]
    return torch.logsumexp(scores, dim=dim) - torch.log(
        torch.tensor(K, dtype=scores.dtype, device=scores.device)
    )


def vanilla_tb_residual(
    logZ: Tensor,
    masked_sum_log_probs: Tensor,
    safe_log_rewards: Tensor,
    max_residual: float,
) -> Tensor:
    """Vanilla TB residual in float32: logZ + sum(log_pi) - log(R).

    Inputs have their masks applied and finite log rewards.
    """
    del max_residual
    return logZ.float() + masked_sum_log_probs.float() - safe_log_rewards.float()


def tb_loss(
    log_probs: Tensor,
    loss_mask: Tensor,
    log_rewards: Tensor,
    logZ: nn.Parameter,
    *,
    max_residual: float = TB_MAX_RESIDUAL_DEFAULT,
    ref_log_probs: Tensor | None = None,
    ref_weight: float = 1.0,
    normalize_by_length: bool = False,
) -> Tensor:
    """Trajectory Balance loss: (logZ + sum(masked_log_probs) - log_R)^2.

    Returns scalar (mean over batch).

    Note: default is False here (low-level API boundary). Config-level
    defaults are True -- callers always pass this explicitly from config.
    """
    mask_f = loss_mask.float()
    masked_logprobs = (log_probs.float() * mask_f).sum(dim=-1)  # [B]

    masked_ref_logprobs = None
    if ref_log_probs is not None:
        if ref_log_probs.dim() == 1:
            masked_ref_logprobs = ref_log_probs
        elif ref_log_probs.dim() == 2:
            if ref_log_probs.shape != log_probs.shape:
                raise ValueError(
                    "ref_log_probs must match log_probs shape when 2D: "
                    f"ref_log_probs={tuple(ref_log_probs.shape)} log_probs={tuple(log_probs.shape)}"
                )
            masked_ref_logprobs = (ref_log_probs * mask_f).sum(dim=-1)
        else:
            raise ValueError(
                f"ref_log_probs must be 1D or 2D, got shape {tuple(ref_log_probs.shape)}"
            )

        if ref_weight != 1.0:
            masked_ref_logprobs = masked_ref_logprobs * ref_weight

    if normalize_by_length:
        lengths = mask_f.sum(dim=-1).clamp_min(1.0)
        masked_logprobs = masked_logprobs / lengths
        if masked_ref_logprobs is not None:
            masked_ref_logprobs = masked_ref_logprobs / lengths

    safe_log_rewards = require_finite_log_rewards(log_rewards)

    if masked_ref_logprobs is not None:
        masked_logprobs = masked_logprobs - masked_ref_logprobs

    residual = vanilla_tb_residual(logZ, masked_logprobs, safe_log_rewards, max_residual)

    loss = (residual**2).mean()

    return loss


def compute_action_subtb_core(
    log_pf: Tensor,
    u: Tensor,
    mask: Tensor | None,
    *,
    subtb_lambda: float = SUBTB_LAMBDA_DEFAULT,
    max_residual: float = TB_MAX_RESIDUAL_DEFAULT,
) -> Tensor:
    """Action-boundary SubTB core loss.

    Same delta-cumsum pattern as token-level modified SubTB but on action transitions:
    delta[i] = u[i] + log_pf[i] - u[i+1], where u is boundary potentials [B, K+1].
    """
    if log_pf.dim() != 2:
        raise ValueError(f"log_pf must be 2D [B, K], got shape={tuple(log_pf.shape)}")
    if u.dim() != 2:
        raise ValueError(f"u must be 2D [B, K+1], got shape={tuple(u.shape)}")
    if u.shape[0] != log_pf.shape[0] or u.shape[1] != log_pf.shape[1] + 1:
        raise ValueError(
            "u shape must be [B, K+1] for log_pf [B, K]: "
            f"log_pf={tuple(log_pf.shape)} u={tuple(u.shape)}"
        )
    if mask is not None and mask.shape != log_pf.shape:
        raise ValueError(
            "mask must match log_pf shape when provided: "
            f"mask={tuple(mask.shape)} log_pf={tuple(log_pf.shape)}"
        )

    B, K = log_pf.shape
    device = log_pf.device
    dtype = log_pf.dtype

    if K == 0:
        return graph_connected_zero(u, log_pf)

    transition_mask = (
        mask.to(device=device, dtype=torch.bool)
        if mask is not None
        else torch.ones(B, K, dtype=torch.bool, device=device)
    )
    mask_f = transition_mask.float()

    safe_log_pf = sanitize_finite(log_pf, -max_residual)
    safe_u = sanitize_finite(u, -max_residual)

    delta = safe_u[:, :-1] + safe_log_pf - safe_u[:, 1:]
    delta = torch.where(transition_mask, delta, torch.zeros_like(delta))

    idxs = torch.arange(K, device=device, dtype=torch.long).expand(B, K)
    sort_keys = (~transition_mask).to(torch.long) * K + idxs
    perm = sort_keys.argsort(dim=1)
    compact_delta = delta.gather(1, perm)
    compact_mask = transition_mask.gather(1, perm).to(dtype=dtype)

    zeros = torch.zeros(B, 1, device=device, dtype=dtype)
    delta_cumsum = torch.cat([zeros, compact_delta], dim=1).cumsum(dim=1)
    mask_cumsum = torch.cat([zeros, compact_mask], dim=1).cumsum(dim=1)

    has_valid = (mask_f.sum(dim=1) > 0).float().unsqueeze(1)

    batch_loss = torch.tensor(0.0, device=device, dtype=dtype)
    total_weight = torch.tensor(0.0, device=device, dtype=dtype)

    # O(K^2) in compute and memory; acceptable for typical action counts (K < 20)
    for subtraj_len in range(1, K + 1):
        residual = delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len]

        mask_sum = mask_cumsum[:, subtraj_len:] - mask_cumsum[:, :-subtraj_len]
        valid_mask = (mask_sum >= subtraj_len - 0.5).float() * has_valid

        residual = residual * valid_mask

        weight = subtb_lambda ** (subtraj_len - 1)
        batch_loss = batch_loss + weight * (residual**2).sum()
        total_weight = total_weight + weight * valid_mask.sum()

    return batch_loss / total_weight.clamp(min=1.0)


def _compute_delta_and_cumsums(
    log_probs: Tensor,
    eos_logprobs: Tensor,
    log_rewards: Tensor,
    mask: Tensor,
    max_residual: float,
    terminal_is_eos: Tensor | None = None,
    eos_log_rewards: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Shared delta computation for both loop and vectorized SubTB.

    Returns (delta_cumsum [B, T+1], mask_cumsum [B, T+1], mask_f [B, T], has_valid [B, 1]).
    """
    B, T = log_probs.shape
    device = log_probs.device
    dtype = log_probs.dtype

    mask_f = mask.float() if mask.dtype != torch.float32 else mask

    if log_rewards.dim() == 1:
        log_rewards_2d = log_rewards.unsqueeze(1).expand(B, T)
    else:
        log_rewards_2d = log_rewards

    safe_log_rewards = require_finite_log_rewards(log_rewards_2d)

    valid = mask_f > 0
    idxs = torch.arange(T, device=device, dtype=torch.long).expand(B, T)
    last_valid_idx = torch.where(valid, idxs, 0).max(dim=1).values
    traj_reward = safe_log_rewards.gather(1, last_valid_idx.unsqueeze(1))

    # Locate the next generated token, skipping observations and padding.
    next_indices = torch.where(valid, idxs, T).flip(1).cummin(dim=1).values.flip(1)
    next_indices = torch.cat([next_indices[:, 1:], idxs.new_full((B, 1), T)], dim=1)
    potentials = -eos_logprobs
    if eos_log_rewards is not None:
        if eos_log_rewards.shape != log_probs.shape:
            raise ValueError(
                "eos_log_rewards must match log_probs shape: "
                f"eos_log_rewards={tuple(eos_log_rewards.shape)} log_probs={tuple(log_probs.shape)}"
            )
        potentials = potentials + require_finite_log_rewards(
            eos_log_rewards.to(device=device, dtype=dtype)
        )
    elif terminal_is_eos is not None:
        if terminal_is_eos.shape != (B,) or terminal_is_eos.dtype != torch.bool:
            raise ValueError("terminal_is_eos must be a boolean tensor with shape [batch]")
        stopping_reward = torch.zeros_like(potentials).scatter(
            1,
            last_valid_idx.unsqueeze(1),
            traj_reward * terminal_is_eos.to(device=device).unsqueeze(1),
        )
        potentials = potentials + stopping_reward
    next_potential = potentials.gather(1, next_indices.clamp(max=T - 1))
    next_term = torch.where(next_indices < T, -next_potential, -traj_reward)
    delta = log_probs + potentials + next_term

    delta = torch.where(valid, delta, torch.zeros_like(delta))

    # Compact valid generated transitions per row so subtrajectories and lambda
    # weights are indexed by generated transitions rather than padded width.
    sort_keys = (~valid).to(torch.long) * T + idxs
    perm = sort_keys.argsort(dim=1)
    compact_delta = delta.gather(1, perm)
    compact_mask = valid.gather(1, perm).to(dtype=dtype)

    zeros = torch.zeros(B, 1, device=device, dtype=dtype)
    delta_cumsum = torch.cat([zeros, compact_delta], dim=1).cumsum(dim=1)
    mask_cumsum = torch.cat([zeros, compact_mask], dim=1).cumsum(dim=1)

    has_valid = (mask_f.sum(dim=1) > 0).float().unsqueeze(1)

    return delta_cumsum, mask_cumsum, mask_f, has_valid


def _subtb_loop(
    delta_cumsum: Tensor,
    mask_cumsum: Tensor,
    has_valid: Tensor,
    T: int,
    subtb_lambda: float,
    max_residual: float,
) -> Tensor:
    """O(T^2) loop-based SubTB residual computation (reference implementation)."""
    per_row_numerators, per_row_denominators = _subtb_loop_terms(
        delta_cumsum,
        mask_cumsum,
        has_valid,
        T,
        subtb_lambda,
        max_residual,
    )
    return per_row_numerators.sum() / per_row_denominators.sum().clamp(min=1.0)


def _subtb_loop_terms(
    delta_cumsum: Tensor,
    mask_cumsum: Tensor,
    has_valid: Tensor,
    T: int,
    subtb_lambda: float,
    max_residual: float,
) -> tuple[Tensor, Tensor]:
    """Loop-based per-row weighted sums and weights for SubTB."""
    del max_residual
    device = delta_cumsum.device
    dtype = delta_cumsum.dtype

    batch_size = delta_cumsum.shape[0]
    per_row_numerators = torch.zeros(batch_size, device=device, dtype=dtype)
    per_row_denominators = torch.zeros(batch_size, device=device, dtype=dtype)

    for subtraj_len in range(1, T + 1):
        residual = delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len]

        mask_sum = mask_cumsum[:, subtraj_len:] - mask_cumsum[:, :-subtraj_len]
        valid_mask = (mask_sum >= subtraj_len - 0.5).float() * has_valid

        residual = residual * valid_mask

        weight = subtb_lambda ** (subtraj_len - 1)
        per_row_numerators = per_row_numerators + weight * (residual**2).sum(dim=1)
        per_row_denominators = per_row_denominators + weight * valid_mask.sum(dim=1)

    return per_row_numerators, per_row_denominators


# element-count guard: each intermediate [B, T+1, T+1] tensor at fp32 costs
# B*(T+1)^2*4 bytes; cap at ~64MB per tensor
_MAX_VECTORIZE_ELEMENTS = 4 * 1024 * 1024  # ~16M elements ≈ 64MB at fp32


def _subtb_vectorized(
    delta_cumsum: Tensor,
    mask_cumsum: Tensor,
    has_valid: Tensor,
    T: int,
    subtb_lambda: float,
    max_residual: float,
) -> Tensor:
    """Vectorized SubTB: O(T^2) memory but no Python loop.

    Computes all pairwise sub-trajectory residuals via broadcasting,
    then weights by lambda^(L-1) using the upper-triangular structure.
    """
    per_row_numerators, per_row_denominators = _subtb_vectorized_terms(
        delta_cumsum,
        mask_cumsum,
        has_valid,
        T,
        subtb_lambda,
        max_residual,
    )
    return per_row_numerators.sum() / per_row_denominators.sum().clamp(min=1.0)


def _subtb_vectorized_terms(
    delta_cumsum: Tensor,
    mask_cumsum: Tensor,
    has_valid: Tensor,
    T: int,
    subtb_lambda: float,
    max_residual: float,
) -> tuple[Tensor, Tensor]:
    """Vectorized per-row weighted sums and weights for SubTB."""
    del max_residual
    B, Tp1 = delta_cumsum.shape  # Tp1 = T + 1
    device = delta_cumsum.device
    dtype = delta_cumsum.dtype

    # all_residuals[b, i, j] = delta_cumsum[b, j] - delta_cumsum[b, i]
    # sub-trajectory of length L starting at i corresponds to (i, i+L)
    all_residuals = delta_cumsum[:, None, :] - delta_cumsum[:, :, None]  # [B, Tp1, Tp1]

    # mask sums for validity
    all_mask_sums = mask_cumsum[:, None, :] - mask_cumsum[:, :, None]  # [B, Tp1, Tp1]

    # length matrix: lengths[i, j] = j - i
    indices = torch.arange(Tp1, device=device, dtype=dtype)
    lengths = indices[None, :] - indices[:, None]  # [Tp1, Tp1]

    # upper triangular: only j > i (length >= 1)
    upper_mask = lengths >= 1  # [Tp1, Tp1]

    # lambda weights: weight for length L is lambda^(L-1)
    # clamp exponent to 0 before raising to avoid 0.0**-1 = inf on diagonal/lower triangle
    safe_exponent = (lengths - 1).clamp(min=0)
    weights = subtb_lambda**safe_exponent  # [Tp1, Tp1]
    weights = weights * upper_mask.float()

    # validity: mask_sum >= length - 0.5 and has_valid
    valid = (all_mask_sums >= (lengths.unsqueeze(0) - 0.5)) & upper_mask.unsqueeze(0)
    valid_f = valid.float() * has_valid.unsqueeze(-1)  # [B, Tp1, Tp1]

    # weighted squared residuals
    weighted_sq = weights.unsqueeze(0) * (all_residuals**2) * valid_f
    weighted_valid = weights.unsqueeze(0) * valid_f

    per_row_numerators = weighted_sq.sum(dim=(1, 2))
    per_row_denominators = weighted_valid.sum(dim=(1, 2))

    return per_row_numerators, per_row_denominators


def compute_modified_subtb_core_terms(
    log_probs: Tensor,
    eos_logprobs: Tensor,
    log_rewards: Tensor,
    mask: Tensor,
    *,
    subtb_lambda: float = SUBTB_LAMBDA_DEFAULT,
    max_residual: float = TB_MAX_RESIDUAL_DEFAULT,
    _force_loop: bool = False,
    terminal_is_eos: Tensor | None = None,
    eos_log_rewards: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return per-trajectory weighted sums and weight totals for modified SubTB."""
    B, T = log_probs.shape

    delta_cumsum, mask_cumsum, mask_f, has_valid = _compute_delta_and_cumsums(
        log_probs,
        eos_logprobs,
        log_rewards,
        mask,
        max_residual,
        terminal_is_eos,
        eos_log_rewards,
    )
    del mask_f

    estimated_elements = B * (T + 1) ** 2
    use_vectorized = not _force_loop and estimated_elements <= _MAX_VECTORIZE_ELEMENTS
    if not use_vectorized:
        return _subtb_loop_terms(
            delta_cumsum,
            mask_cumsum,
            has_valid,
            T,
            subtb_lambda,
            max_residual,
        )

    return _subtb_vectorized_terms(
        delta_cumsum,
        mask_cumsum,
        has_valid,
        T,
        subtb_lambda,
        max_residual,
    )


def compute_modified_subtb_core(
    log_probs: Tensor,
    eos_logprobs: Tensor,
    log_rewards: Tensor,
    mask: Tensor,
    *,
    subtb_lambda: float = SUBTB_LAMBDA_DEFAULT,
    max_residual: float = TB_MAX_RESIDUAL_DEFAULT,
    _force_loop: bool = False,
    terminal_is_eos: Tensor | None = None,
    eos_log_rewards: Tensor | None = None,
) -> Tensor:
    """Modified SubTB with optional per-prefix stop rewards or research surrogate.

    When ``eos_log_rewards`` is provided with shape ``[B, T]``, each prefix ``t``
    has consistent state potential ``eos_log_rewards[b, t] - eos_logprobs[b, t]``,
    enforcing exact terminal-reward proportionality across prefixes shared by
    different trajectories.

    When ``eos_log_rewards`` is omitted, this function computes a prefix-regularized
    research surrogate where non-stopping prefixes use unit early-stopping reward
    (potential ``-log pi(EOS | s)``), the stopping prefix under ``terminal_is_eos``
    has potential ``log R - log pi(EOS | s)``, and the terminal state has potential
    ``log R``. Because a shared prefix's potential then depends on whether the
    sampled trajectory stops there, omitting ``eos_log_rewards`` does not enforce
    ordinary terminal-reward proportionality across variable-length prefixes.

    loss = sum_L lambda^(L-1) * (cumsum(delta)[t+L] - cumsum(delta)[t])^2

    Returns scalar (weighted mean over valid sub-trajectories).

    When _force_loop=True, always uses the loop implementation (for testing).
    When _force_loop=False (default), uses vectorized for T <= 1024, loop otherwise.
    """
    per_row_numerators, per_row_denominators = compute_modified_subtb_core_terms(
        log_probs,
        eos_logprobs,
        log_rewards,
        mask,
        subtb_lambda=subtb_lambda,
        max_residual=max_residual,
        _force_loop=_force_loop,
        terminal_is_eos=terminal_is_eos,
        eos_log_rewards=eos_log_rewards,
    )
    return per_row_numerators.sum() / per_row_denominators.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Research loss variants (TB+KL, FlowRL, VarGrad)
# ---------------------------------------------------------------------------


def _prepare_tb_inputs(
    log_probs: Tensor,
    response_mask: Tensor,
    reward_temp: float,
    length_normalize: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Shared preamble for research TB loss functions.

    Validates reward_temp, computes mask/lengths, and returns the
    (optionally length-normalized) trajectory log-prob sum.

    Returns (mask_f, response_lengths, traj_log_prob).
    """
    if reward_temp <= 0:
        raise ValueError(f"reward_temp must be positive, got {reward_temp}")

    mask_f = response_mask.float()
    response_lengths = mask_f.sum(dim=-1).clamp(min=1.0)

    traj_log_prob = (log_probs * mask_f).sum(dim=-1)
    if length_normalize:
        traj_log_prob = traj_log_prob / response_lengths

    return mask_f, response_lengths, traj_log_prob


def estimate_log_partition_vargrad(
    log_probs: Tensor,
    log_rewards: Tensor,
    response_mask: Tensor,
    ref_log_probs: Tensor | None = None,
    reward_temp: float = 1.0,
    length_normalize: bool = False,
) -> Tensor:
    """Batch logZ estimator via log-mean-exp (no learned parameter).

    Estimates logZ per prompt from a batch of K responses, avoiding the need
    for a learned logZ parameter. Works best with K >= 4 responses per prompt.

    When length_normalize=True, log_probs (and ref_log_probs) are divided by
    response length before computing the score. The resulting logZ is in the
    normalized scale and must be used with similarly normalized residuals.

    Formula: log(mean(exp(score_k))), computed as logsumexp - log(K) for stability:
        logZ(x) = logsumexp_k(r(y_k)/beta - log pi_theta(y_k|x) + log pi_ref(y_k|x)) - log(K)
    """
    mask_f, response_lengths, traj_log_prob = _prepare_tb_inputs(
        log_probs,
        response_mask,
        reward_temp,
        length_normalize,
    )

    score = log_rewards / reward_temp - traj_log_prob

    if ref_log_probs is not None:
        ref_traj_log_prob = (ref_log_probs * mask_f).sum(dim=-1)
        if length_normalize:
            ref_traj_log_prob = ref_traj_log_prob / response_lengths
        score = score + ref_traj_log_prob

    logZ = _log_mean_exp(score, dim=0)
    return logZ


def compute_tb_loss_with_kl(
    log_probs: Tensor,
    log_rewards: Tensor,
    response_mask: Tensor,
    logZ: Tensor,
    ref_log_probs: Tensor | None = None,
    length_normalize: bool = True,
    reward_temp: float = 1.0,
    max_residual: float = TB_MAX_RESIDUAL_DEFAULT,
) -> tuple[Tensor, dict[str, float]]:
    """TB loss with optional KL regularization (FlowRL/TBA combined)."""
    mask_f, response_lengths, traj_log_prob = _prepare_tb_inputs(
        log_probs,
        response_mask,
        reward_temp,
        length_normalize,
    )

    ref_contribution = torch.zeros_like(traj_log_prob)
    if ref_log_probs is not None:
        ref_traj_log_prob = (ref_log_probs * mask_f).sum(dim=-1)
        if length_normalize:
            ref_traj_log_prob = ref_traj_log_prob / response_lengths
        ref_contribution = ref_traj_log_prob

    scaled_rewards = log_rewards / reward_temp if reward_temp != 1.0 else log_rewards
    residual = logZ + traj_log_prob - scaled_rewards - ref_contribution
    residual = residual.clamp(-max_residual, max_residual)
    loss = (residual**2).mean()

    metrics = {
        "tb_loss": loss.item(),
        "mean_residual": residual.mean().item(),
        "std_residual": residual.std().item(),
        "mean_traj_log_prob": traj_log_prob.mean().item(),
        "mean_response_length": response_lengths.mean().item(),
    }
    if ref_log_probs is not None:
        metrics["mean_ref_log_prob"] = ref_contribution.mean().item()

    return loss, metrics


def compute_flowrl_loss(
    log_probs: Tensor,
    log_rewards: Tensor,
    response_mask: Tensor,
    logZ: Tensor,
    ref_log_probs: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    length_normalize: bool = True,
    reward_temp: float = 15.0,
    max_residual: float = TB_MAX_RESIDUAL_DEFAULT,
    use_importance_weights: bool = False,
) -> tuple[Tensor, dict[str, float]]:
    """Full FlowRL loss (arXiv:2509.15207) with importance weights."""
    B = log_probs.shape[0]
    mask_f, response_lengths, traj_log_prob = _prepare_tb_inputs(
        log_probs,
        response_mask,
        reward_temp,
        length_normalize,
    )

    ref_contribution = torch.zeros_like(traj_log_prob)
    if ref_log_probs is not None:
        ref_traj = (ref_log_probs * mask_f).sum(dim=-1)
        if length_normalize:
            ref_traj = ref_traj / response_lengths
        ref_contribution = ref_traj

    scaled_rewards = log_rewards / reward_temp if reward_temp != 1.0 else log_rewards
    residual = logZ + traj_log_prob - scaled_rewards - ref_contribution
    residual = residual.clamp(-max_residual, max_residual)
    squared_residual = residual**2

    if use_importance_weights and old_log_probs is not None:
        old_traj_log_prob = (old_log_probs * mask_f).sum(dim=-1)
        log_ratio = (log_probs * mask_f).sum(dim=-1) - old_traj_log_prob
        log_ratio = log_ratio.clamp(-5.0, 5.0)
        weights = torch.exp(log_ratio).detach()
    else:
        if use_importance_weights and old_log_probs is None:
            logger.warning("use_importance_weights=True but old_log_probs not provided")
        weights = torch.ones(B, device=log_probs.device, dtype=log_probs.dtype)

    loss = (weights * squared_residual).mean()

    metrics = {
        "flowrl_loss": loss.item(),
        "mean_residual": residual.mean().item(),
        "mean_is_weight": weights.mean().item(),
        "reward_temp": reward_temp,
    }
    return loss, metrics


def compute_vargrad_tb_loss(
    log_probs: Tensor,
    log_rewards: Tensor,
    response_mask: Tensor,
    ref_log_probs: Tensor | None = None,
    length_normalize: bool = True,
    reward_temp: float = 1.0,
    max_residual: float = TB_MAX_RESIDUAL_DEFAULT,
    prompt_ids: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """TB loss with batch-estimated logZ via log-mean-exp (no learned parameter)."""
    B = log_probs.shape[0]
    mask_f, response_lengths, traj_log_prob_normed = _prepare_tb_inputs(
        log_probs,
        response_mask,
        reward_temp,
        length_normalize,
    )

    ref_contribution_normed = torch.zeros_like(traj_log_prob_normed)
    if ref_log_probs is not None:
        ref_traj = (ref_log_probs * mask_f).sum(dim=-1)
        ref_contribution_normed = ref_traj / response_lengths if length_normalize else ref_traj

    # logZ estimation must use the same scale (normalized vs raw) as the
    # residual to avoid a length-dependent bias. When length_normalize=True,
    # use the per-token-averaged log-probs and ref contributions.
    logZ_log_prob = traj_log_prob_normed
    logZ_ref = ref_contribution_normed

    if prompt_ids is not None:
        unique_prompts = prompt_ids.unique()
        logZ_per_sample = torch.zeros(B, device=log_probs.device, dtype=log_probs.dtype)
        for pid in unique_prompts:
            group_mask = prompt_ids == pid
            group_scores = (
                log_rewards[group_mask] / reward_temp
                - logZ_log_prob[group_mask]
                + logZ_ref[group_mask]
            )
            group_logZ = _log_mean_exp(group_scores, dim=0)
            logZ_per_sample[group_mask] = group_logZ
        logZ = logZ_per_sample
    else:
        scores = log_rewards / reward_temp - logZ_log_prob + logZ_ref
        logZ = _log_mean_exp(scores, dim=0)
        logZ = logZ.expand(B)

    logZ_detached = logZ.detach()
    scaled_rewards = log_rewards / reward_temp if reward_temp != 1.0 else log_rewards
    residual = logZ_detached + traj_log_prob_normed - scaled_rewards - ref_contribution_normed
    residual = residual.clamp(-max_residual, max_residual)
    loss = (residual**2).mean()

    metrics = {
        "vargrad_loss": loss.item(),
        "estimated_logZ": logZ_detached.mean().item(),
        "mean_residual": residual.mean().item(),
        "std_residual": residual.std(unbiased=False).item(),
    }
    return loss, metrics
