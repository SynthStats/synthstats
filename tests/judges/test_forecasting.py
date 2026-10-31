"""Tests for ForecastingJudge."""

from __future__ import annotations

import math

import pytest

from synthstats.core.types import Trajectory
from synthstats.judges.forecasting import (
    ForecastingJudge,
    _crps_energy,
    _mae,
    _point_forecast,
    _rmse,
    _scale_metric_to_reward,
)


def _stub_trajectory() -> Trajectory:
    return Trajectory.empty_stub(messages=[])


def _perfect_samples(targets: list[int], n_samples: int = 50) -> list[list[float]]:
    """Samples that all equal the target exactly (perfect forecast)."""
    return [[float(t) for t in targets] for _ in range(n_samples)]


def _flat_samples(value: float, horizon: int, n_samples: int = 50) -> list[list[float]]:
    return [[value] * horizon for _ in range(n_samples)]


# ------------------------------------------------------------------
# Unit: helper functions
# ------------------------------------------------------------------


class TestPointForecast:
    def test_even_samples(self):
        samples = [[10.0, 20.0], [12.0, 22.0]]
        pt = _point_forecast(samples)
        assert pt[0] == pytest.approx(11.0)
        assert pt[1] == pytest.approx(21.0)

    def test_odd_samples(self):
        samples = [[10.0], [12.0], [14.0]]
        pt = _point_forecast(samples)
        assert pt[0] == pytest.approx(12.0)

    def test_single_sample(self):
        samples = [[7.5, 8.5]]
        pt = _point_forecast(samples)
        assert pt == pytest.approx([7.5, 8.5])

    def test_empty_returns_empty(self):
        assert _point_forecast([]) == []


class TestMAE:
    def test_perfect_forecast(self):
        point = [10.0, 20.0, 30.0]
        targets = [10, 20, 30]
        assert _mae(point, targets) == pytest.approx(0.0)

    def test_known_error(self):
        point = [10.0, 10.0]
        targets = [15, 5]
        assert _mae(point, targets) == pytest.approx(5.0)

    def test_empty_returns_nan(self):
        assert math.isnan(_mae([], []))


class TestRMSE:
    def test_perfect_forecast(self):
        point = [5.0, 10.0]
        targets = [5, 10]
        assert _rmse(point, targets) == pytest.approx(0.0)

    def test_known_error(self):
        # errors are [3, 4], rmse = sqrt((9+16)/2) = sqrt(12.5)
        point = [3.0, 4.0]
        targets = [0, 0]
        assert _rmse(point, targets) == pytest.approx(math.sqrt(12.5))

    def test_empty_returns_nan(self):
        assert math.isnan(_rmse([], []))


class TestCRPSEnergy:
    def test_perfect_deterministic_forecast_zero_crps(self):
        targets = [100, 120, 110]
        samples = _perfect_samples(targets, n_samples=20)
        crps = _crps_energy(samples, targets)
        assert crps == pytest.approx(0.0, abs=1e-6)

    def test_crps_increases_with_error(self):
        targets = [100] * 5
        good_samples = _flat_samples(100.0, horizon=5, n_samples=30)
        bad_samples = _flat_samples(120.0, horizon=5, n_samples=30)
        assert _crps_energy(good_samples, targets) < _crps_energy(bad_samples, targets)

    def test_crps_with_spread_is_better_than_wrong_point(self):
        targets = [100] * 5
        # deterministic bad forecast
        bad_point = _flat_samples(80.0, horizon=5, n_samples=30)
        # wide spread that covers the truth
        spread = [[80.0 + i * 2 for _ in range(5)] for i in range(30)]
        assert _crps_energy(spread, targets) < _crps_energy(bad_point, targets)

    def test_empty_samples_returns_nan(self):
        assert math.isnan(_crps_energy([], [100]))

    def test_empty_targets_returns_nan(self):
        assert math.isnan(_crps_energy([[100.0]], []))


class TestScaleMetricToReward:
    def test_zero_error_gives_one(self):
        assert _scale_metric_to_reward(0.0, scale=10.0) == pytest.approx(1.0)

    def test_one_scale_gives_exp_minus_one(self):
        assert _scale_metric_to_reward(10.0, scale=10.0) == pytest.approx(math.exp(-1.0))

    def test_nonfinite_gives_zero(self):
        assert _scale_metric_to_reward(float("nan"), scale=1.0) == 0.0
        assert _scale_metric_to_reward(float("inf"), scale=1.0) == 0.0

    def test_negative_gives_zero(self):
        assert _scale_metric_to_reward(-1.0, scale=1.0) == 0.0


# ------------------------------------------------------------------
# ForecastingJudge
# ------------------------------------------------------------------


class TestForecastingJudge:
    def test_zero_reward_when_no_targets(self):
        judge = ForecastingJudge()
        reward = judge.score(
            task_name="forecast",
            trajectory=_stub_trajectory(),
            artifacts={},
        )
        assert reward.total == 0.0

    def test_perfect_forecast_gives_high_reward(self):
        targets = [100, 110, 105, 115, 120, 130, 125, 110, 115, 120]
        samples = _perfect_samples(targets, n_samples=50)
        judge = ForecastingJudge()
        reward = judge.score(
            task_name="forecast",
            trajectory=_stub_trajectory(),
            artifacts={
                "forecast_targets": targets,
                "forecast_samples": samples,
            },
        )
        assert reward.total > 0.9
        assert reward.components["mae"] == pytest.approx(0.0, abs=1e-6)
        assert reward.components["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert reward.components["crps"] == pytest.approx(0.0, abs=1e-6)

    def test_poor_forecast_gives_low_reward(self):
        targets = [100] * 10
        # very wrong samples (off by 200 — well beyond scale)
        samples = _flat_samples(300.0, horizon=10, n_samples=50)
        judge = ForecastingJudge(mae_scale=15.0, rmse_scale=20.0, crps_scale=12.0)
        reward = judge.score(
            task_name="forecast",
            trajectory=_stub_trajectory(),
            artifacts={
                "forecast_targets": targets,
                "forecast_samples": samples,
            },
        )
        assert reward.total < 0.1

    def test_reward_in_zero_one_range(self):
        targets = [100, 110, 90, 105, 115, 120, 108, 112, 119, 100]
        # moderately good samples
        samples = [[t + (i % 5) * 2 - 4 for t in targets] for i in range(40)]
        judge = ForecastingJudge()
        reward = judge.score(
            task_name="forecast",
            trajectory=_stub_trajectory(),
            artifacts={
                "forecast_targets": targets,
                "forecast_samples": samples,
            },
        )
        assert 0.0 <= reward.total <= 1.0

    def test_reward_components_populated(self):
        targets = [100] * 5
        samples = _flat_samples(100.0, horizon=5, n_samples=20)
        judge = ForecastingJudge()
        reward = judge.score(
            task_name="forecast",
            trajectory=_stub_trajectory(),
            artifacts={
                "forecast_targets": targets,
                "forecast_samples": samples,
            },
        )
        for key in ("mae", "rmse", "crps", "mae_factor", "rmse_factor", "crps_factor"):
            assert key in reward.components

    def test_no_samples_and_no_program_gives_zero(self):
        judge = ForecastingJudge()
        reward = judge.score(
            task_name="forecast",
            trajectory=_stub_trajectory(),
            artifacts={"forecast_targets": [100] * 10},
        )
        assert reward.total == 0.0

    def test_judge_exported_from_judges_init(self):
        from synthstats.judges import ForecastingJudge as FJ

        assert FJ is ForecastingJudge
