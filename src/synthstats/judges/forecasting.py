"""ForecastingJudge scores probabilistic time-series forecasts.

Metrics:
  - MAE  (mean absolute error, point forecast = predictive median)
  - RMSE (root mean squared error)
  - CRPS (continuous ranked probability score, averaged over horizon)

The judge expects forecast evaluation artifacts:
  - ``forecast_samples``: list[list[float]] shape (n_samples, horizon)
  - ``forecast_targets``: list[int] shape (horizon,); observed targets
  - ``forecast_horizon``: int

The CRPS computation evaluates the empirical predictive distribution:
  CRPS(F, y) = E[|X - y|] - 0.5 * E[|X - X'|]
Both expectations use the same ensemble, including diagonal pairs.
"""

from __future__ import annotations

import json
import logging
import math
import re
import subprocess
from collections.abc import Sequence
from typing import Any

from synthstats.core.reward_semantics import annotate_reward_info
from synthstats.core.types import Reward, Trajectory

logger = logging.getLogger(__name__)

_FORECAST_RUNNER_TEMPLATE = """
import json as _json
import sys as _sys

{user_code}

# --- synthstats: extract forecast samples ---
_samples = None

# try explicit __FORECAST_SAMPLES__ print first (handled by parent process)
# if not printed, look for forecast_samples variable in globals
if "forecast_samples" in dir():
    import numpy as _np
    _arr = _np.asarray(forecast_samples)
    if _arr.ndim == 2:
        print("__FORECAST_SAMPLES__:" + _json.dumps(_arr.tolist()))
    elif _arr.ndim == 1:
        # single draw — wrap in outer list
        print("__FORECAST_SAMPLES__:" + _json.dumps([_arr.tolist()]))
"""

_FORECAST_EXTRACT_RE = re.compile(r"^__FORECAST_SAMPLES__:(.+)$", re.MULTILINE)


def _extract_forecast_samples(stdout: str) -> list[list[float]] | None:
    """Pull the __FORECAST_SAMPLES__ sentinel from subprocess stdout."""
    match = _FORECAST_EXTRACT_RE.search(stdout)
    if match is None:
        return None
    try:
        data = json.loads(match.group(1))
        if isinstance(data, list) and data:
            if isinstance(data[0], list):
                return data
            # 1-D list (single sample) — wrap
            return [data]
    except (json.JSONDecodeError, IndexError, TypeError):
        pass
    return None


def _run_forecast_program(
    program: str, timeout: float = 120.0
) -> tuple[list[list[float]] | None, str]:
    """Execute a forecast program in a subprocess and extract samples."""
    from synthstats.executors.isolated_python import run_isolated_python
    from synthstats.executors.pymc_sandbox import check_code_safety

    is_safe, error = check_code_safety(program)
    if not is_safe:
        return None, f"safety check failed: {error}"

    code = _FORECAST_RUNNER_TEMPLATE.format(user_code=program)
    try:
        result = run_isolated_python(code, timeout=timeout)
        stdout, stderr = result.stdout, result.stderr
    except (subprocess.SubprocessError, OSError) as exc:
        return None, str(exc)

    samples = _extract_forecast_samples(stdout)
    if samples is None and stderr:
        logger.debug("Forecast program stderr: %s", stderr[:400])
    return samples, stderr


def _point_forecast(samples: list[list[float]]) -> list[float]:
    """Predictive median per horizon step."""
    n_samples = len(samples)
    n_horizon = len(samples[0]) if samples else 0
    medians = []
    for t in range(n_horizon):
        col = sorted(s[t] for s in samples)
        mid = n_samples // 2
        if n_samples % 2 == 0:
            medians.append((col[mid - 1] + col[mid]) / 2.0)
        else:
            medians.append(float(col[mid]))
    return medians


def _mae(point: Sequence[float], targets: Sequence[int | float]) -> float:
    if not point or not targets:
        return float("nan")
    n = min(len(point), len(targets))
    return sum(abs(point[t] - targets[t]) for t in range(n)) / n


def _rmse(point: Sequence[float], targets: Sequence[int | float]) -> float:
    if not point or not targets:
        return float("nan")
    n = min(len(point), len(targets))
    return math.sqrt(sum((point[t] - targets[t]) ** 2 for t in range(n)) / n)


def _crps_energy(
    samples: Sequence[Sequence[float]],
    targets: Sequence[int | float],
) -> float:
    """Energy-form CRPS averaged over forecast horizon.

    CRPS(F, y) = E_F[|X - y|] - 0.5 * E_F[|X - X'|]
    """
    n_samples = len(samples)
    n_horizon = min(len(samples[0]) if samples else 0, len(targets))
    if n_horizon == 0 or n_samples == 0:
        return float("nan")

    total_crps = 0.0
    for t in range(n_horizon):
        y = float(targets[t])
        col = [s[t] for s in samples]

        # E[|X - y|]
        e_abs_y = sum(abs(x - y) for x in col) / n_samples

        # Sum of unordered pair distances after sorting: O(n log n).
        col.sort()
        pair_sum = sum((2 * i - n_samples + 1) * value for i, value in enumerate(col))
        total_crps += e_abs_y - pair_sum / (n_samples * n_samples)

    return total_crps / n_horizon


def _scale_metric_to_reward(metric: float, scale: float) -> float:
    """Map a lower-is-better metric to (0, 1] via exp(-metric / scale)."""
    if not math.isfinite(metric) or metric < 0:
        return 0.0
    return math.exp(-metric / scale)


class ForecastingJudge:
    """Reward signal for time-series probabilistic forecasting.

    Computes MAE, RMSE, and CRPS against held-out target actuals, then
    scalarises to a reward in (0, 1].

    The final reward is a geometric mean of the three metric-based
    factors, each mapped to (0, 1] via ``exp(-metric / scale)``.

    ``mae_scale``, ``rmse_scale`` and ``crps_scale`` set the metric scales.
    """

    def __init__(
        self,
        mae_scale: float = 15.0,
        rmse_scale: float = 20.0,
        crps_scale: float = 12.0,
        timeout: float = 120.0,
    ) -> None:
        self.mae_scale = mae_scale
        self.rmse_scale = rmse_scale
        self.crps_scale = crps_scale
        self.timeout = timeout

    def score(self, *, task_name: str, trajectory: Trajectory, artifacts: dict[str, Any]) -> Reward:
        del task_name, trajectory

        targets: list[int] = artifacts.get("forecast_targets", [])
        precomputed_samples: Any = artifacts.get("forecast_samples")
        program: str = artifacts.get("program", "")

        if not targets:
            return self._zero_reward("no forecast_targets in artifacts")

        horizon = artifacts.get("forecast_horizon", len(targets))
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon != len(targets):
            return self._zero_reward("forecast_horizon must equal the target horizon")
        try:
            finite_targets = all(math.isfinite(float(value)) for value in targets)
        except (TypeError, ValueError, OverflowError):
            finite_targets = False
        if not finite_targets:
            return self._zero_reward("forecast targets must be finite numbers")

        # resolve samples: prefer precomputed, fall back to running the program
        samples: list[list[float]] | None = None
        exec_error: str = ""

        if precomputed_samples is not None:
            try:
                samples = [list(map(float, row)) for row in precomputed_samples]
            except (TypeError, ValueError) as exc:
                logger.warning("Could not parse precomputed forecast_samples: %s", exc)
                return self._zero_reward("forecast_samples must be a numeric matrix")

        if samples is None and program:
            samples, exec_error = _run_forecast_program(program, timeout=self.timeout)

        if samples is None:
            return self._zero_reward(f"no forecast samples available; exec_error={exec_error!r}")

        if not samples or any(
            len(row) != horizon or not all(math.isfinite(value) for value in row) for row in samples
        ):
            return self._zero_reward("forecast_samples must be finite and cover the full horizon")

        point = _point_forecast(samples)
        mae = _mae(point, targets)
        rmse = _rmse(point, targets)
        crps = _crps_energy(samples, targets)

        mae_factor = _scale_metric_to_reward(mae, self.mae_scale)
        rmse_factor = _scale_metric_to_reward(rmse, self.rmse_scale)
        crps_factor = _scale_metric_to_reward(crps, self.crps_scale)

        # geometric mean of the three factors
        log_total = (
            math.log(max(mae_factor, 1e-12))
            + math.log(max(rmse_factor, 1e-12))
            + math.log(max(crps_factor, 1e-12))
        ) / 3.0
        total = math.exp(log_total)

        return Reward(
            total=total,
            components={
                "mae": mae,
                "rmse": rmse,
                "crps": crps,
                "mae_factor": mae_factor,
                "rmse_factor": rmse_factor,
                "crps_factor": crps_factor,
            },
            info=annotate_reward_info(
                {
                    "mae": mae,
                    "rmse": rmse,
                    "crps": crps,
                    "n_samples": len(samples),
                    "horizon": len(targets),
                },
                source="forecasting",
            ),
        )

    def _zero_reward(self, reason: str) -> Reward:
        logger.warning("ForecastingJudge: zero reward — %s", reason)
        return Reward(
            total=0.0,
            components={},
            info=annotate_reward_info({"reason": reason}, source="forecasting"),
        )


__all__ = [
    "ForecastingJudge",
    "_crps_energy",
    "_mae",
    "_rmse",
    "_point_forecast",
    "_scale_metric_to_reward",
]
