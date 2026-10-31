"""Runner metric extraction helpers."""

from __future__ import annotations

import torch


def extract_trainer_metrics(trainer: object | None) -> dict[str, float]:
    """Collect numeric trainer metrics plus logZ into the run result."""
    metrics: dict[str, float] = {}
    if trainer is None:
        return metrics

    all_metrics = getattr(trainer, "all_metrics", None)
    if isinstance(all_metrics, dict):
        for key, value in all_metrics.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int | float):
                metrics[str(key)] = float(value)

    logz = getattr(trainer, "logZ", None)
    if isinstance(logz, torch.Tensor):
        metrics["logZ"] = float(logz.item())

    return metrics


__all__ = ["extract_trainer_metrics"]
