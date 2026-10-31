"""Invalid likelihood evidence cannot earn a reward."""

import pytest

from synthstats.core.types import Trajectory
from synthstats.judges.likelihood import LikelihoodJudge


@pytest.mark.parametrize("matrix", [[], [-1.0, -2.0], [[-1.0, -2.0]]])
@pytest.mark.parametrize("mode", ["strict", "nonfinite"])
def test_invalid_log_likelihood_is_rejected(matrix, mode):
    reward = LikelihoodJudge(loo_gate_mode=mode).score(
        task_name="test",
        trajectory=Trajectory.empty_stub([]),
        artifacts={"runtime": "lazyppl", "log_lik_matrix": matrix},
    )
    assert reward.total == 0.0
    assert reward.info["loo_status"] == "computation_failed"
    assert reward.info["would_reject_strict"] is True


@pytest.mark.parametrize("matrix", [[], [[-1.0, -2.0]]])
def test_valid_precomputed_fallback_still_scores(matrix):
    reward = LikelihoodJudge().score(
        task_name="test",
        trajectory=Trajectory.empty_stub([]),
        artifacts={
            "diagnostics": {"warning": False, "good_k": 0.7, "n_over_good_k": 0},
            "log_lik_matrix": matrix,
            "elpd": -2.0,
        },
    )
    assert reward.total == pytest.approx(0.8187307530779818)
    assert reward.info["elpd_source"] == "precomputed_fallback"


@pytest.mark.parametrize("diagnostics", [None, {}, {"warning": False}])
def test_strict_judge_rejects_missing_diagnostics(diagnostics):
    reward = LikelihoodJudge().score(
        task_name="test",
        trajectory=Trajectory.empty_stub([]),
        artifacts={"language": "pymc", "elpd": 2.0, "diagnostics": diagnostics},
    )
    assert reward.total == 0.0
    assert reward.info["would_reject_strict"] is True
