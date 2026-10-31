"""Required judge failures retain their weight in a composite reward."""

import pytest

from synthstats.core.types import Trajectory
from synthstats.judges.composite import CompositeJudge
from synthstats.judges.likelihood import LikelihoodJudge


class UnavailableJudge:
    def score(self, **kwargs):
        raise RuntimeError("scoring service unavailable")


def test_required_failed_factor_cannot_increase_product_reward():
    reward = CompositeJudge([(LikelihoodJudge(beta=0.0), 0.5), (UnavailableJudge(), 0.5)]).score(
        task_name="test",
        trajectory=Trajectory.empty_stub([]),
        artifacts={
            "elpd": 1.0,
            "diagnostics": {"warning": False, "good_k": 0.7, "n_over_good_k": 0},
        },
    )
    # sqrt(1 * 1e-12), the same floor as a required zero-scoring factor.
    assert reward.total == pytest.approx(1e-6)
    assert reward.factors["UnavailableJudge"] == 0.0
    assert reward.info["factor_weights"]["UnavailableJudge"] == 0.5
