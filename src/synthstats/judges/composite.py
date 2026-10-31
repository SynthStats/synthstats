"""Weighted combination of multiple judges."""

from __future__ import annotations

import logging
import math
from typing import Any

from omegaconf import DictConfig, ListConfig

from synthstats.core.judge import Judge
from synthstats.core.reward_semantics import annotate_reward_info
from synthstats.core.types import Reward, Trajectory

logger = logging.getLogger(__name__)


class CompositeJudge:
    """Weighted combination of multiple judges. Accepts (judge, weight) tuples or dicts."""

    def __init__(
        self,
        judges: list[tuple[Judge, float]] | list[dict[str, Any]],
        *,
        scalarization_mode: str = "weighted_log_product",
        min_factor: float = 1e-12,
    ):
        self.judges = self._normalize_judges(judges)
        self.scalarization_mode = scalarization_mode
        self.min_factor = min_factor
        supported = {"weighted_sum", "weighted_log_product"}
        if self.scalarization_mode not in supported:
            raise ValueError(
                f"Unsupported scalarization_mode={scalarization_mode!r}; "
                "expected 'weighted_sum' or 'weighted_log_product'"
            )
        self._warned_weight_sum = False

    def _normalize_judges(
        self,
        judges: list[tuple[Judge, float]] | list[dict[str, Any]],
    ) -> list[tuple[Judge, float]]:
        normalized: list[tuple[Judge, float]] = []
        candidates = list(judges) if isinstance(judges, ListConfig) else judges
        for spec in candidates:
            judge_obj: Any
            weight_val: Any

            if isinstance(spec, tuple | list):
                if len(spec) != 2:
                    raise ValueError(
                        "Tuple/list judge specs must be (judge, weight) pairs in CompositeJudge"
                    )
                judge_obj, weight_val = spec
            elif isinstance(spec, dict) or isinstance(spec, DictConfig):
                judge_obj = spec.get("judge")
                weight_val = spec.get("weight", 1.0)
            else:
                raise TypeError(
                    "CompositeJudge judges entries must be (judge, weight) pairs "
                    "or {'judge': ..., 'weight': ...} dicts"
                )

            if judge_obj is None:
                raise ValueError("CompositeJudge received a judge spec with judge=None")
            normalized.append((judge_obj, float(weight_val)))
        return normalized

    def _maybe_warn_weight_sum(self) -> None:
        if self._warned_weight_sum or self.scalarization_mode != "weighted_log_product":
            return
        weight_sum = sum(w for _, w in self.judges)
        if abs(weight_sum - 1.0) > 1e-6:
            logger.warning(
                "weighted_log_product weights sum to %.4f, not 1.0; "
                "reward magnitude will differ from a geometric mean",
                weight_sum,
            )
            self._warned_weight_sum = True

    def score(self, *, task_name: str, trajectory: Trajectory, artifacts: dict[str, Any]) -> Reward:
        if not self.judges:
            return Reward(
                total=0.0,
                components={},
                factors={},
                scalarization=self.scalarization_mode,
                info=annotate_reward_info(
                    {"scalarization_mode": self.scalarization_mode, "factor_weights": {}},
                    source="composite",
                ),
            )

        self._maybe_warn_weight_sum()

        total = 0.0
        weighted_log_total = 0.0
        components: dict[str, float] = {}
        factors: dict[str, float] = {}
        factor_weights: dict[str, float] = {}
        errors: dict[str, str] = {}
        label_counts: dict[str, int] = {}

        active_weight_sum = 0.0
        for judge, weight in self.judges:
            judge_name = type(judge).__name__
            seen = label_counts.get(judge_name, 0)
            label_counts[judge_name] = seen + 1
            label = judge_name if seen == 0 else f"{judge_name}#{seen + 1}"

            try:
                sub_reward = judge.score(
                    task_name=task_name, trajectory=trajectory, artifacts=artifacts
                )
            except (ValueError, TypeError, RuntimeError, ArithmeticError, KeyError) as exc:
                logger.warning(
                    "Judge %s failed (%s), zeroing component",
                    label,
                    type(exc).__name__,
                    exc_info=True,
                )
                errors[label] = f"{type(exc).__name__}: {exc}"
                sub_reward = Reward(total=0.0, components={}, info={})

            raw_total = float(sub_reward.total)
            components[label] = raw_total
            factors[label] = raw_total
            factor_weights[label] = float(weight)
            active_weight_sum += weight

            if self.scalarization_mode == "weighted_sum":
                total += weight * raw_total
            else:
                weighted_log_total += weight * math.log(max(raw_total, self.min_factor))

        if self.scalarization_mode == "weighted_log_product" and active_weight_sum > 0:
            total = math.exp(weighted_log_total)
        elif self.scalarization_mode == "weighted_log_product":
            total = 0.0

        return Reward(
            total=total,
            components=components,
            factors=factors,
            scalarization=self.scalarization_mode,
            info=annotate_reward_info(
                {
                    "scalarization_mode": self.scalarization_mode,
                    "factor_weights": factor_weights,
                    **({"errors": errors} if errors else {}),
                },
                source="composite",
            ),
        )
