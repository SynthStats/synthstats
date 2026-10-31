"""GFlowNetTrainer: thin SkyRL trainer adapter for GFlowNet losses.

The native policy worker owns log Z, its optimizer group and its checkpoint.
The standalone trainer core remains available without the distributed SDK.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from synthstats.core.constants import LOGZ_LR_DEFAULT
from synthstats.core.import_helpers import try_import_symbol
from synthstats.train.config_validation import validate_gfn_algorithm_config
from synthstats.train.gfn_core import (
    GFlowNetTrainerCore,
    LogZModule,
    resolve_logz_params,
)
from synthstats.train.lineage import (
    _LINEAGE_SEQUENCE_ALIASES,
    apply_lineage_batching,
    compute_lineage_batch_order,
    compute_lineage_loss_weight,
    compute_lineage_loss_weights,
    extract_numeric_rollout_metrics,
    extract_rollout_metric_sequences,
)

logger = logging.getLogger(__name__)


def _build_lineage_rows(
    rollout_metrics: Mapping[str, Any],
    num_rows: int,
) -> list[dict[str, Any]] | None:
    """Pack per-trajectory sequence metrics into row dicts that slice/concat with trajectories."""
    if num_rows <= 0:
        return None
    row_sequences: dict[str, list[Any]] = {}
    for key, value in rollout_metrics.items():
        if isinstance(value, list | tuple) and len(value) == num_rows:
            row_sequences[str(key)] = list(value)
        elif hasattr(value, "dim") and hasattr(value, "detach") and value.dim() == 1:
            seq = value.detach().cpu().tolist()
            if isinstance(seq, list) and len(seq) == num_rows:
                row_sequences[str(key)] = seq
    if not row_sequences:
        return None
    return [
        {key: deepcopy(seq[row_idx]) for key, seq in row_sequences.items()}
        for row_idx in range(num_rows)
    ]


def _reconstruct_rollout_metrics_from_rows(
    lineage_rows: Sequence[Mapping[str, Any]],
    *,
    base_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild rollout_metrics sequences and counts after dynamic-sampling row transforms."""
    metrics: dict[str, Any] = (
        deepcopy(dict(base_metrics)) if isinstance(base_metrics, Mapping) else {}
    )
    if not lineage_rows:
        return metrics

    keys = [str(key) for key in lineage_rows[0].keys()]
    for key in keys:
        if all(isinstance(row, Mapping) and key in row for row in lineage_rows):
            metrics[key] = [deepcopy(row[key]) for row in lineage_rows]

    for count_key, flag_key in _LINEAGE_SEQUENCE_ALIASES.items():
        seq = metrics.get(flag_key)
        if isinstance(seq, list):
            metrics[count_key] = sum(
                1
                for item in seq
                if (
                    isinstance(item, int | float)
                    and not isinstance(item, bool)
                    and float(item) > 0.0
                )
                or item is True
            )

    if "n_success" in metrics:
        metrics["n_success"] = len(lineage_rows)
    return metrics


def _normalize_group_uids_for_boundaries(
    uids: list[str],
    *,
    train_batch_size: int,
    n_samples_per_prompt: int,
    is_stepwise: bool,
) -> list[str]:
    """Ensure replaced prompt groups remain distinct contiguous slots for boundary computation."""
    if (
        is_stepwise
        or train_batch_size <= 0
        or n_samples_per_prompt <= 0
        or len(uids) != train_batch_size * n_samples_per_prompt
    ):
        return uids
    group_uids = [uids[g * n_samples_per_prompt] for g in range(train_batch_size)]
    if len(set(group_uids)) == train_batch_size:
        return uids
    return [f"{uids[idx]}__slot{idx // n_samples_per_prompt}" for idx in range(len(uids))]


_RayPPOTrainer = try_import_symbol(
    "RayPPOTrainer",
    "skyrl.train.trainer",
)
if _RayPPOTrainer is None:
    logger.debug("SkyRL not installed; GFlowNetTrainer will run standalone")

try:
    if _RayPPOTrainer is None:
        raise ImportError("SkyRL not available")

    RayPPOTrainer = _RayPPOTrainer

    class GFlowNetTrainer(RayPPOTrainer, GFlowNetTrainerCore):  # type: ignore[misc, valid-type]
        """Use the stock trainer with the policy-owned TB normalizer."""

        _pending_rollout_metrics: dict[str, Any] | None = None

        def __init__(self, cfg: Any, *args: Any, **kwargs: Any) -> None:
            self._verify_gfn_config(cfg)
            super().__init__(cfg, *args, **kwargs)

        def _verify_gfn_config(self, config: Any) -> None:
            validate_gfn_algorithm_config(config)

        async def generate(self, input_batch: Any) -> Any:
            generator_output = await self.generator.generate(input_batch)
            raw_rollout_metrics = generator_output.get("rollout_metrics")
            response_ids = (
                generator_output.get("response_ids")
                if isinstance(generator_output, Mapping)
                else None
            )
            num_rows = len(response_ids) if isinstance(response_ids, list) else 0
            if isinstance(raw_rollout_metrics, Mapping):
                self._pending_rollout_metrics = deepcopy(dict(raw_rollout_metrics))
                lineage_rows = _build_lineage_rows(raw_rollout_metrics, num_rows)
                if lineage_rows is not None and isinstance(generator_output, dict):
                    generator_output["_lineage_rows"] = lineage_rows
                self.all_metrics.update(
                    {
                        k: v
                        for k, v in raw_rollout_metrics.items()
                        if isinstance(v, int | float) and not isinstance(v, bool)
                    }
                )
            else:
                self._pending_rollout_metrics = None
                if isinstance(generator_output, dict):
                    generator_output.pop("_lineage_rows", None)
            generator_output.pop("rollout_metrics", None)
            from skyrl.train.utils.trainer_utils import validate_generator_output

            validate_generator_output(
                len(input_batch["prompts"]),
                generator_output,
                step_wise=self.cfg.generator.step_wise_trajectories,
                routes_expected=self.cfg.generator.inference_engine.enable_return_routed_experts,
            )
            return generator_output

        def handle_dynamic_sampling(
            self,
            generator_output: Any,
            uids: list[str],
        ) -> tuple[Any, list[str], bool]:
            if isinstance(generator_output, dict) and "_lineage_rows" not in generator_output:
                response_ids = generator_output.get("response_ids")
                num_rows = len(response_ids) if isinstance(response_ids, list) else 0
                source_metrics = generator_output.get("rollout_metrics")
                if not isinstance(source_metrics, Mapping):
                    source_metrics = getattr(self, "_pending_rollout_metrics", None)
                if isinstance(source_metrics, Mapping):
                    lineage_rows = _build_lineage_rows(source_metrics, num_rows)
                    if lineage_rows is not None:
                        generator_output["_lineage_rows"] = lineage_rows

            processed_output, processed_uids, keep_sampling = super().handle_dynamic_sampling(
                generator_output,
                uids,
            )
            if (
                not keep_sampling
                and isinstance(processed_output, Mapping)
                and isinstance(processed_output.get("_lineage_rows"), list)
            ):
                self._pending_rollout_metrics = _reconstruct_rollout_metrics_from_rows(
                    processed_output["_lineage_rows"],
                    base_metrics=getattr(self, "_pending_rollout_metrics", None),
                )
            return processed_output, processed_uids, keep_sampling

        def convert_to_training_input(self, generator_output: Any, uids: list[str]) -> Any:
            rollout_metrics: dict[str, Any] | None = None
            pending = getattr(self, "_pending_rollout_metrics", None)
            base_metrics: dict[str, Any] = {}
            if isinstance(pending, Mapping):
                base_metrics.update(deepcopy(dict(pending)))
            if isinstance(generator_output, Mapping) and isinstance(
                generator_output.get("rollout_metrics"), Mapping
            ):
                base_metrics.update(deepcopy(dict(generator_output["rollout_metrics"])))

            lineage_rows = (
                generator_output.get("_lineage_rows")
                if isinstance(generator_output, Mapping)
                else None
            )
            response_ids = (
                generator_output.get("response_ids")
                if isinstance(generator_output, Mapping)
                else None
            )
            if (
                isinstance(lineage_rows, list)
                and lineage_rows
                and isinstance(response_ids, list)
                and len(lineage_rows) == len(response_ids)
            ):
                rollout_metrics = _reconstruct_rollout_metrics_from_rows(
                    lineage_rows,
                    base_metrics=base_metrics,
                )
            elif base_metrics:
                rollout_metrics = base_metrics
            elif isinstance(getattr(self, "all_metrics", None), Mapping) and (
                extract_rollout_metric_sequences({"rollout_metrics": self.all_metrics})
                or extract_numeric_rollout_metrics({"rollout_metrics": self.all_metrics})
            ):
                rollout_metrics = deepcopy(dict(self.all_metrics))
            self._pending_rollout_metrics = None

            boundary_uids = _normalize_group_uids_for_boundaries(
                uids,
                train_batch_size=int(getattr(self.cfg.trainer, "train_batch_size", 0) or 0),
                n_samples_per_prompt=int(
                    getattr(self.cfg.generator, "n_samples_per_prompt", 1) or 1
                ),
                is_stepwise=bool(getattr(self.cfg.generator, "step_wise_trajectories", False)),
            )
            training_input = super().convert_to_training_input(generator_output, boundary_uids)
            if rollout_metrics is not None:
                metadata = getattr(training_input, "metadata", None)
                if isinstance(metadata, dict):
                    metadata["rollout_metrics"] = rollout_metrics
                if isinstance(getattr(self, "all_metrics", None), dict):
                    self.all_metrics.update(
                        {
                            k: v
                            for k, v in rollout_metrics.items()
                            if isinstance(v, int | float) and not isinstance(v, bool)
                        }
                    )
            return training_input

        def _normalize_advantages(
            self,
            data: Any,
            mini_batch_boundaries: list[tuple[int, int]] | None = None,
            prompt_boundaries: list[tuple[int, int]] | None = None,
        ) -> Any:
            """Keep raw trajectory log-rewards intact and apply lineage controls."""
            del mini_batch_boundaries, prompt_boundaries
            return self.apply_batch_lineage_controls(data)

        def build_models(self, policy_worker, critic_worker, ref_worker):
            if self.cfg.trainer.strategy != "fsdp":
                raise ValueError("The policy-owned TB normalizer requires trainer.strategy=fsdp")
            from synthstats.integrations.skyrl.policy_worker import PolicyWorker

            return super().build_models(PolicyWorker, critic_worker, ref_worker)

    SKYRL_TRAINER_AVAILABLE = True
    logger.debug("GFlowNetTrainer loaded with RayPPOTrainer backend")

except ImportError:

    class GFlowNetTrainer(GFlowNetTrainerCore):  # type: ignore[no-redef]
        """Standalone GFlowNet trainer (no SkyRL).

        Provides logZ management only. Use for local testing or as a mixin base.
        """

        def __init__(
            self,
            config: Any = None,
            *args: Any,
            logZ_init: float = 0.0,
            logZ_lr: float = LOGZ_LR_DEFAULT,
            **kwargs: Any,
        ) -> None:
            if config is not None:
                logZ_init, logZ_lr = resolve_logz_params(config)
            self.init_gfn(logZ_init=logZ_init, logZ_lr=logZ_lr)
            logger.debug("GFlowNetTrainer running without SkyRL")

    SKYRL_TRAINER_AVAILABLE = False


__all__ = [
    "GFlowNetTrainer",
    "LogZModule",
    "SKYRL_TRAINER_AVAILABLE",
    "apply_lineage_batching",
    "compute_lineage_batch_order",
    "compute_lineage_loss_weight",
    "compute_lineage_loss_weights",
    "extract_numeric_rollout_metrics",
    "extract_rollout_metric_sequences",
    "validate_gfn_algorithm_config",
]
