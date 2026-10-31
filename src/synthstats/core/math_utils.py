"""Tensor math utilities."""

from __future__ import annotations

import torch
from torch import Tensor


def sanitize_finite(tensor: Tensor, fallback: float) -> Tensor:
    """Replace NaN/Inf values with a fallback constant."""
    return torch.where(torch.isfinite(tensor), tensor, torch.full_like(tensor, fallback))


def graph_connected_zero(*tensors: Tensor) -> Tensor:
    """Create a gradient-connected zero from input tensors.

    Returns a zero tensor that still participates in the computation graph,
    preventing errors during backward() on degenerate batches.
    """
    if not tensors:
        return torch.tensor(0.0, requires_grad=True)

    zero = tensors[0].sum() * 0.0
    for tensor in tensors[1:]:
        zero = zero + tensor.sum() * 0.0
    if not zero.requires_grad:
        zero = zero.detach().clone().requires_grad_(True)
    return zero


def drop_empty_mask_rows(
    *tensors: Tensor,
    mask: Tensor,
) -> tuple[tuple[Tensor, ...], bool]:
    """Filter out batch rows where mask sums to zero.

    Returns (filtered_tensors, any_dropped).
    """
    row_valid = mask.sum(dim=-1) > 0
    if row_valid.all():
        return tensors, False
    filtered = tuple(t[row_valid] for t in tensors)
    return filtered, True
