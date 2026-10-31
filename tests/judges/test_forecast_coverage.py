"""Forecast rewards require complete finite target coverage."""

import pytest

from synthstats.core.types import Trajectory
from synthstats.judges.forecasting import ForecastingJudge


@pytest.mark.parametrize(
    "samples",
    [
        [[100.0]],
        [],
        [[]],
        [[100.0, 1000.0, 2000.0], [100.0]],
        [[100.0, float("nan"), 2000.0]],
        [[100.0, float("inf"), 2000.0]],
        [[100.0, 1000.0, 2000.0, 3000.0]],
    ],
)
def test_invalid_forecast_cannot_earn_full_horizon_reward(samples):
    result = ForecastingJudge().score(
        task_name="forecast",
        trajectory=Trajectory.empty_stub([]),
        artifacts={
            "forecast_samples": samples,
            "forecast_targets": [100, 1000, 2000],
            "forecast_horizon": 3,
        },
    )
    assert result.total == 0.0
    assert "reason" in result.info


def test_declared_horizon_must_match_targets():
    result = ForecastingJudge().score(
        task_name="forecast",
        trajectory=Trajectory.empty_stub([]),
        artifacts={"forecast_samples": [[100.0]], "forecast_targets": [100], "forecast_horizon": 3},
    )
    assert result.total == 0.0
