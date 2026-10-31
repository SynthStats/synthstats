"""Rollout-lineage helpers for training integrations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from synthstats.core.config_utils import config_get as _config_value
from synthstats.train.config_validation import resolve_algorithm_config

_LINEAGE_SEQUENCE_ALIASES: dict[str, str] = {
    "n_with_trace_id": "has_trace_id",
    "n_with_task_id": "has_task_id",
    "n_with_dataset_row": "has_dataset_row",
    "n_with_shared_generation_record": "has_shared_generation_record",
    "n_with_shared_trial_record": "has_shared_trial_record",
    "n_with_shared_trace_record": "has_shared_trace_record",
    "n_with_verifier_reward": "has_verifier_reward",
}


def _finite_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        candidate = float(value)
        if math.isfinite(candidate):
            return candidate
    return default


def _optional_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        candidate = float(value)
        if math.isfinite(candidate):
            return candidate
    return None


def _numeric_metric_items(metrics: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            out[str(key)] = float(value)
    return out


def _sanitize_metric_label(label: str) -> str:
    compact = []
    last_was_sep = False
    for char in label.strip().lower():
        if char.isalnum():
            compact.append(char)
            last_was_sep = False
            continue
        if not last_was_sep:
            compact.append("_")
            last_was_sep = True
    return "".join(compact).strip("_") or "unknown"


def _provenance_metric_items(metrics: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, prefix in (
        ("reward_evidence_kinds", "reward_evidence_kind"),
        ("reward_truth_owners", "reward_truth_owner"),
    ):
        raw_values = metrics.get(key)
        if not isinstance(raw_values, list | tuple) or not raw_values:
            continue

        normalized: list[str] = []
        valid = True
        for item in raw_values:
            if not isinstance(item, str):
                valid = False
                break
            normalized.append(_sanitize_metric_label(item))
        if not valid or not normalized:
            continue

        total = float(len(normalized))
        counts: dict[str, int] = {}
        for label in normalized:
            counts[label] = counts.get(label, 0) + 1

        for label in sorted(counts):
            count = counts[label]
            out[f"n_{prefix}_{label}"] = float(count)
            out[f"{prefix}_{label}_fraction"] = float(count) / total
    return out


def _rollout_metric_mappings(data: Any) -> list[Mapping[str, Any]]:
    mappings: list[Mapping[str, Any]] = []

    if isinstance(data, Mapping):
        candidate = data.get("rollout_metrics")
        if isinstance(candidate, Mapping):
            mappings.append(candidate)

    for attr_name in ("rollout_metrics", "non_tensor_batch", "metadata", "meta_info", "meta"):
        container = getattr(data, attr_name, None)
        if isinstance(container, Mapping):
            mappings.append(container)
            nested = container.get("rollout_metrics")
            if isinstance(nested, Mapping):
                mappings.append(nested)

    return mappings


def _numeric_metric_sequences(metrics: Mapping[str, Any]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.dim() != 1:
                continue
            out[str(key)] = [float(item) for item in value.detach().cpu().tolist()]
            continue
        if not isinstance(value, list | tuple) or not value:
            continue

        sequence: list[float] = []
        valid = True
        for item in value:
            if isinstance(item, bool):
                sequence.append(float(item))
            elif isinstance(item, int | float):
                candidate = float(item)
                if not math.isfinite(candidate):
                    valid = False
                    break
                sequence.append(candidate)
            else:
                valid = False
                break
        if valid:
            out[str(key)] = sequence
    return out


def extract_numeric_rollout_metrics(data: Any) -> dict[str, float]:
    """Extract numeric rollout metrics from plausible batch containers."""
    for candidate in _rollout_metric_mappings(data):
        numeric = _numeric_metric_items(candidate)
        provenance = _provenance_metric_items(candidate)
        if numeric or provenance:
            return {
                **numeric,
                **provenance,
            }
    return {}


def extract_rollout_metric_sequences(data: Any) -> dict[str, list[float]]:
    """Extract rollout-level numeric sequences from plausible batch containers."""
    for candidate in _rollout_metric_mappings(data):
        sequences = _numeric_metric_sequences(candidate)
        if sequences:
            return sequences
    return {}


def _resolve_lineage_signal_sequence(
    rollout_sequences: Mapping[str, list[float]],
    *,
    metric_name: str,
    denominator_name: str | None,
) -> list[float] | None:
    signal_values = rollout_sequences.get(metric_name)
    if signal_values is None:
        alias = _LINEAGE_SEQUENCE_ALIASES.get(metric_name)
        if alias is not None:
            signal_values = rollout_sequences.get(alias)
    if signal_values is None:
        return None

    if denominator_name:
        denominator_values = rollout_sequences.get(denominator_name)
        if denominator_values is not None and len(denominator_values) == len(signal_values):
            ratios: list[float] = []
            for numerator, denominator in zip(
                signal_values,
                denominator_values,
                strict=True,
            ):
                ratios.append(numerator / denominator if denominator > 0 else 0.0)
            return ratios

    return signal_values


def compute_lineage_loss_weights(data: Any, config: Any) -> list[float] | None:
    """Compute per-rollout loss multipliers from rollout lineage metadata."""
    algorithm = resolve_algorithm_config(config)
    weighting = _config_value(algorithm, "lineage_weighting")
    if weighting is None:
        return None

    metric_name = _config_value(weighting, "metric")
    if not isinstance(metric_name, str) or not metric_name:
        return None

    rollout_sequences = extract_rollout_metric_sequences(data)
    if not rollout_sequences:
        return None

    denominator_name = _config_value(weighting, "denominator_metric")
    resolved_denominator = denominator_name if isinstance(denominator_name, str) else None
    signal_values = _resolve_lineage_signal_sequence(
        rollout_sequences,
        metric_name=metric_name,
        denominator_name=resolved_denominator,
    )
    if signal_values is None:
        return None

    base_weight = _finite_float(_config_value(weighting, "base_weight", 1.0), 1.0)
    scale = _finite_float(_config_value(weighting, "scale", 0.0), 0.0)
    min_weight = _optional_finite_float(_config_value(weighting, "min_weight"))
    max_weight = _optional_finite_float(_config_value(weighting, "max_weight"))

    weights: list[float] = []
    for signal in signal_values:
        weight = base_weight + scale * signal
        if min_weight is not None:
            weight = max(weight, min_weight)
        if max_weight is not None:
            weight = min(weight, max_weight)
        weights.append(weight if math.isfinite(weight) else 1.0)

    return weights or None


def _resolve_lineage_batch_groups(
    data: Any,
    config: Any,
) -> tuple[list[int], int, int, bool] | None:
    algorithm = resolve_algorithm_config(config)
    batching = _config_value(algorithm, "lineage_batching")
    if batching is None:
        return None

    metric_name = _config_value(batching, "metric")
    if not isinstance(metric_name, str) or not metric_name:
        return None

    rollout_sequences = extract_rollout_metric_sequences(data)
    if not rollout_sequences:
        return None

    denominator_name = _config_value(batching, "denominator_metric")
    resolved_denominator = denominator_name if isinstance(denominator_name, str) else None
    signal_values = _resolve_lineage_signal_sequence(
        rollout_sequences,
        metric_name=metric_name,
        denominator_name=resolved_denominator,
    )
    if signal_values is None or len(signal_values) < 2:
        return None

    threshold = _finite_float(_config_value(batching, "signal_threshold", 0.0), 0.0)
    positive_first = bool(_config_value(batching, "positive_first", True))

    positive_indices = [index for index, signal in enumerate(signal_values) if signal > threshold]
    negative_indices = [index for index, signal in enumerate(signal_values) if signal <= threshold]
    if not positive_indices or not negative_indices:
        return None

    order = (
        positive_indices + negative_indices
        if positive_first
        else negative_indices + positive_indices
    )
    if order == list(range(len(signal_values))):
        return None

    return order, len(positive_indices), len(negative_indices), positive_first


def compute_lineage_batch_order(data: Any, config: Any) -> list[int] | None:
    """Compute a stable verifier-first or heuristic-first rollout order."""
    grouping = _resolve_lineage_batch_groups(data, config)
    if grouping is None:
        return None
    order, _, _, _ = grouping
    return order


_NON_REORDERABLE_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "policy_mini_batch_boundaries",
        "critic_mini_batch_boundaries",
        "policy_prompt_boundaries",
        "critic_prompt_boundaries",
        "metrics",
    }
)


def _reorder_batch_value(
    value: Any,
    *,
    order: list[int],
    batch_size: int,
) -> Any:
    if isinstance(value, torch.Tensor):
        if value.dim() > 0 and value.shape[0] == batch_size:
            index = torch.tensor(order, dtype=torch.long, device=value.device)
            return value.index_select(0, index)
        return value

    if hasattr(value, "tensors") and callable(getattr(value, "__getitem__", None)):
        try:
            if len(value) == batch_size:
                return value[order]
        except Exception:
            return value

    if isinstance(value, dict):
        for key, child in list(value.items()):
            if key in _NON_REORDERABLE_METADATA_KEYS:
                continue
            value[key] = _reorder_batch_value(
                child,
                order=order,
                batch_size=batch_size,
            )
        return value

    if isinstance(value, list):
        if len(value) == batch_size:
            return [value[index] for index in order]
        return value

    if isinstance(value, tuple):
        if len(value) == batch_size:
            return tuple(value[index] for index in order)
        return value

    return value


def _iter_reorderable_attribute_names(data: Any) -> list[str]:
    try:
        attr_names = set(vars(data))
    except TypeError:
        attr_names = set()
    for attr_name in dir(data):
        if attr_name.startswith("__") or attr_name in attr_names:
            continue
        try:
            value = getattr(data, attr_name)
        except AttributeError:
            continue
        if callable(value):
            continue
        attr_names.add(attr_name)
    return sorted(attr_names)


def apply_lineage_batching(data: Any, config: Any) -> dict[str, float]:
    """Reorder batch-shaped fields into verifier-first or heuristic-first groups."""
    grouping = _resolve_lineage_batch_groups(data, config)
    if grouping is None:
        return {}

    order, positive_count, negative_count, positive_first = grouping
    batch_size = len(order)

    if isinstance(data, dict):
        metadata = getattr(data, "metadata", None)
        pad_size = int(metadata.get("pad_size", 0) or 0) if isinstance(metadata, Mapping) else 0
        tensor_order = (
            order + list(range(batch_size, batch_size + pad_size)) if pad_size > 0 else order
        )
        tensor_batch_size = len(tensor_order)
        for key, value in list(data.items()):
            data[key] = _reorder_batch_value(
                value,
                order=tensor_order,
                batch_size=tensor_batch_size,
            )
        for attr_name in ("metadata", "rollout_metrics", "non_tensor_batch", "meta_info", "meta"):
            attr_val = getattr(data, attr_name, None)
            if attr_val is not None:
                setattr(
                    data,
                    attr_name,
                    _reorder_batch_value(
                        attr_val,
                        order=order,
                        batch_size=batch_size,
                    ),
                )
    else:
        for attr_name in _iter_reorderable_attribute_names(data):
            setattr(
                data,
                attr_name,
                _reorder_batch_value(
                    getattr(data, attr_name),
                    order=order,
                    batch_size=batch_size,
                ),
            )

    return {
        "lineage_batch_reordered": 1.0,
        "lineage_batch_positive_count": float(positive_count),
        "lineage_batch_negative_count": float(negative_count),
        "lineage_batch_positive_first": 1.0 if positive_first else 0.0,
    }


def _compute_lineage_loss_weight_scalar(data: Any, config: Any) -> float:
    """Compute the batch-level loss multiplier from aggregate metrics."""
    algorithm = resolve_algorithm_config(config)
    weighting = _config_value(algorithm, "lineage_weighting")
    if weighting is None:
        return 1.0

    metric_name = _config_value(weighting, "metric")
    if not isinstance(metric_name, str) or not metric_name:
        return 1.0

    rollout_metrics = extract_numeric_rollout_metrics(data)
    base_weight = _finite_float(_config_value(weighting, "base_weight", 1.0), 1.0)
    scale = _finite_float(_config_value(weighting, "scale", 0.0), 0.0)
    missing_signal = _finite_float(_config_value(weighting, "missing_signal", 0.0), 0.0)

    signal = missing_signal
    numerator = rollout_metrics.get(metric_name)
    denominator_name = _config_value(weighting, "denominator_metric")
    if isinstance(denominator_name, str) and denominator_name:
        denominator = rollout_metrics.get(denominator_name)
        if numerator is not None and denominator is not None and denominator > 0:
            signal = numerator / denominator
    elif numerator is not None:
        signal = numerator

    weight = base_weight + scale * signal
    min_weight = _optional_finite_float(_config_value(weighting, "min_weight"))
    if min_weight is not None:
        weight = max(weight, min_weight)
    max_weight = _optional_finite_float(_config_value(weighting, "max_weight"))
    if max_weight is not None:
        weight = min(weight, max_weight)
    if not math.isfinite(weight):
        return 1.0
    return weight


def compute_lineage_loss_weight(data: Any, config: Any) -> float:
    """Compute a batch-level loss multiplier from rollout lineage metrics."""
    per_rollout_weights = compute_lineage_loss_weights(data, config)
    if per_rollout_weights:
        return sum(per_rollout_weights) / len(per_rollout_weights)
    return _compute_lineage_loss_weight_scalar(data, config)


__all__ = [
    "apply_lineage_batching",
    "compute_lineage_batch_order",
    "compute_lineage_loss_weight",
    "compute_lineage_loss_weights",
    "extract_numeric_rollout_metrics",
    "extract_rollout_metric_sequences",
]
