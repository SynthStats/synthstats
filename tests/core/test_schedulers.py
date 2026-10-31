import pytest

from synthstats.core.schedulers import RewardTemperatureSchedule


def test_linear_endpoints():
    s = RewardTemperatureSchedule(start=1.0, end=0.1, horizon=100)
    assert s.get(0) == pytest.approx(1.0)
    assert s.get(100) == pytest.approx(0.1)


def test_linear_midpoint():
    s = RewardTemperatureSchedule(start=1.0, end=0.0, horizon=100)
    assert s.get(50) == pytest.approx(0.5)


def test_clamps_beyond_horizon():
    s = RewardTemperatureSchedule(start=1.0, end=0.1, horizon=100)
    assert s.get(200) == pytest.approx(0.1)
    assert s.get(9999) == pytest.approx(0.1)


def test_cosine_monotonic_and_reaches_end():
    s = RewardTemperatureSchedule(start=1.0, end=0.1, horizon=100, mode="cosine")
    vals = [s.get(i) for i in range(101)]
    assert vals[0] == pytest.approx(1.0)
    for a, b in zip(vals, vals[1:], strict=False):
        assert a >= b - 1e-9
    assert vals[-1] == pytest.approx(0.1)


def test_horizon_zero_returns_end():
    s = RewardTemperatureSchedule(start=1.0, end=0.5, horizon=0)
    assert s.get(0) == pytest.approx(0.5)
    assert s.get(10) == pytest.approx(0.5)


def test_scale_reward():
    s = RewardTemperatureSchedule(start=2.0, end=2.0, horizon=100)
    assert s.scale_reward(10.0, step=0) == pytest.approx(5.0)
