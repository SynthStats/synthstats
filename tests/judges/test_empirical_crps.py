"""Exact CRPS of the empirical predictive distribution."""

import pytest

from synthstats.judges.forecasting import _crps_energy


def test_crps_uses_the_same_distribution_in_both_terms():
    samples = [[-100.0]] * 100 + [[100.0]] * 100 + [[0.0]] * 800
    # E|X| = 20 and E|X-X'| = 36, including diagonal pairs.
    assert _crps_energy(samples, [0]) == pytest.approx(2.0)
    assert _crps_energy(samples[::-1], [0]) == pytest.approx(2.0)


def test_two_draw_empirical_crps_includes_diagonal_pairs():
    # E|X| = 1; the four distances are 0, 2, 2, 0.
    assert _crps_energy([[-1.0], [1.0]], [0]) == pytest.approx(0.5)
