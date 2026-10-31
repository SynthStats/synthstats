from __future__ import annotations

import logging
import math

import pytest

from synthstats.core.types import Message, Reward, Trajectory
from synthstats.judges.composite import CompositeJudge


def _traj() -> Trajectory:
    return Trajectory(
        messages=[Message(role="user", content="test")],
        token_ids=[],
        token_logprobs=[],
        loss_mask=[],
        reward=Reward(total=0.0, components={}, info={}),
    )


class _FixedJudge:
    def __init__(self, score: float):
        self._score = score

    def score(self, *, task_name: str, trajectory: Trajectory, artifacts: dict) -> Reward:
        return Reward(total=self._score, components={}, info={})


class TestNormalizeJudges:
    def test_dict_spec_with_explicit_weight(self):
        j = _FixedJudge(2.0)
        composite = CompositeJudge([{"judge": j, "weight": 0.5}], scalarization_mode="weighted_sum")
        reward = composite.score(task_name="t", trajectory=_traj(), artifacts={})
        assert reward.total == pytest.approx(0.5 * 2.0)

    def test_dict_spec_default_weight_is_one(self):
        j = _FixedJudge(3.0)
        composite = CompositeJudge([{"judge": j}], scalarization_mode="weighted_sum")
        reward = composite.score(task_name="t", trajectory=_traj(), artifacts={})
        assert reward.total == pytest.approx(3.0)

    def test_omegaconf_dictconfig_spec(self):
        j = _FixedJudge(5.0)
        spec = {"judge": j, "weight": 0.2}
        composite = CompositeJudge([spec], scalarization_mode="weighted_sum")
        reward = composite.score(task_name="t", trajectory=_traj(), artifacts={})
        assert reward.total == pytest.approx(1.0)

    def test_omegaconf_listconfig_of_tuples(self):
        j = _FixedJudge(4.0)
        specs = [(j, 0.5)]
        composite = CompositeJudge(specs, scalarization_mode="weighted_sum")
        reward = composite.score(task_name="t", trajectory=_traj(), artifacts={})
        assert reward.total == pytest.approx(2.0)

    def test_list_pair_treated_like_tuple(self):
        j = _FixedJudge(6.0)
        composite = CompositeJudge([[j, 0.25]], scalarization_mode="weighted_sum")
        reward = composite.score(task_name="t", trajectory=_traj(), artifacts={})
        assert reward.total == pytest.approx(1.5)


class TestNormalizeJudgesErrors:
    def test_tuple_wrong_length_raises(self):
        with pytest.raises(ValueError, match="must be.*pairs"):
            CompositeJudge([(_FixedJudge(1.0),)])

    def test_tuple_three_elements_raises(self):
        with pytest.raises(ValueError, match="must be.*pairs"):
            CompositeJudge([(_FixedJudge(1.0), 0.5, "extra")])

    def test_none_judge_in_dict_raises(self):
        with pytest.raises(ValueError, match="judge=None"):
            CompositeJudge([{"judge": None, "weight": 1.0}])

    def test_invalid_spec_type_raises(self):
        with pytest.raises(TypeError, match="must be.*pairs.*dicts"):
            CompositeJudge(["not_a_valid_spec"])

    def test_unsupported_scalarization_mode_raises(self):
        with pytest.raises(ValueError, match="Unsupported scalarization_mode"):
            CompositeJudge([], scalarization_mode="product")


class TestDuplicateJudgeLabels:
    def test_two_judges_same_class_get_numbered_labels(self):
        j1 = _FixedJudge(2.0)
        j2 = _FixedJudge(3.0)
        composite = CompositeJudge(
            [(j1, 0.5), (j2, 0.5)],
            scalarization_mode="weighted_sum",
        )
        reward = composite.score(task_name="t", trajectory=_traj(), artifacts={})

        assert "_FixedJudge" in reward.components
        assert "_FixedJudge#2" in reward.components
        assert reward.components["_FixedJudge"] == pytest.approx(2.0)
        assert reward.components["_FixedJudge#2"] == pytest.approx(3.0)

    def test_three_judges_same_class(self):
        judges = [(_FixedJudge(float(i)), 1.0) for i in range(3)]
        composite = CompositeJudge(judges, scalarization_mode="weighted_sum")
        reward = composite.score(task_name="t", trajectory=_traj(), artifacts={})

        assert "_FixedJudge" in reward.components
        assert "_FixedJudge#2" in reward.components
        assert "_FixedJudge#3" in reward.components


class TestWeightedLogProduct:
    def test_geometric_mean_of_equal_scores(self):
        composite = CompositeJudge(
            [(_FixedJudge(4.0), 0.5), (_FixedJudge(4.0), 0.5)],
        )
        reward = composite.score(task_name="t", trajectory=_traj(), artifacts={})
        # exp(0.5*log(4) + 0.5*log(4)) = 4.0
        assert reward.total == pytest.approx(4.0)

    def test_geometric_mean_of_different_scores(self):
        composite = CompositeJudge(
            [(_FixedJudge(1.0), 0.5), (_FixedJudge(9.0), 0.5)],
        )
        reward = composite.score(task_name="t", trajectory=_traj(), artifacts={})
        # exp(0.5*log(1) + 0.5*log(9)) = exp(0.5*log(9)) = 3.0
        assert reward.total == pytest.approx(3.0)

    def test_unequal_weights_shift_toward_heavier(self):
        composite = CompositeJudge(
            [(_FixedJudge(1.0), 0.9), (_FixedJudge(100.0), 0.1)],
        )
        reward = composite.score(task_name="t", trajectory=_traj(), artifacts={})
        expected = math.exp(0.9 * math.log(1.0) + 0.1 * math.log(100.0))
        assert reward.total == pytest.approx(expected)

    def test_nonpositive_score_clamped_to_min_factor(self):
        min_f = 1e-8
        composite = CompositeJudge(
            [(_FixedJudge(0.0), 0.5), (_FixedJudge(4.0), 0.5)],
            min_factor=min_f,
        )
        reward = composite.score(task_name="t", trajectory=_traj(), artifacts={})
        expected = math.exp(0.5 * math.log(min_f) + 0.5 * math.log(4.0))
        assert reward.total == pytest.approx(expected)
        assert math.isfinite(reward.total)


class TestWeightWarning:
    def test_warns_when_weights_dont_sum_to_one(self, caplog):
        composite = CompositeJudge(
            [(_FixedJudge(2.0), 0.3), (_FixedJudge(2.0), 0.3)],
        )
        with caplog.at_level(logging.WARNING, logger="synthstats.judges.composite"):
            composite.score(task_name="t", trajectory=_traj(), artifacts={})

        assert any("sum to" in r.message for r in caplog.records)

    def test_warns_only_once(self, caplog):
        composite = CompositeJudge(
            [(_FixedJudge(2.0), 0.4), (_FixedJudge(2.0), 0.4)],
        )
        with caplog.at_level(logging.WARNING, logger="synthstats.judges.composite"):
            composite.score(task_name="t", trajectory=_traj(), artifacts={})
            composite.score(task_name="t", trajectory=_traj(), artifacts={})

        weight_warnings = [r for r in caplog.records if "sum to" in r.message]
        assert len(weight_warnings) == 1

    def test_no_warning_when_weights_sum_to_one(self, caplog):
        composite = CompositeJudge(
            [(_FixedJudge(2.0), 0.5), (_FixedJudge(2.0), 0.5)],
        )
        with caplog.at_level(logging.WARNING, logger="synthstats.judges.composite"):
            composite.score(task_name="t", trajectory=_traj(), artifacts={})

        weight_warnings = [r for r in caplog.records if "sum to" in r.message]
        assert len(weight_warnings) == 0

    def test_no_warning_for_weighted_sum_mode(self, caplog):
        composite = CompositeJudge(
            [(_FixedJudge(2.0), 0.3), (_FixedJudge(2.0), 0.3)],
            scalarization_mode="weighted_sum",
        )
        with caplog.at_level(logging.WARNING, logger="synthstats.judges.composite"):
            composite.score(task_name="t", trajectory=_traj(), artifacts={})

        weight_warnings = [r for r in caplog.records if "sum to" in r.message]
        assert len(weight_warnings) == 0


class TestRewardMetadata:
    def test_reward_contains_factors_and_scalarization(self):
        composite = CompositeJudge(
            [(_FixedJudge(2.0), 0.7)],
            scalarization_mode="weighted_sum",
        )
        reward = composite.score(task_name="t", trajectory=_traj(), artifacts={})

        assert reward.factors == {"_FixedJudge": 2.0}
        assert reward.scalarization == "weighted_sum"
        assert reward.info["factor_weights"] == {"_FixedJudge": 0.7}

    def test_empty_judges_reward_metadata(self):
        composite = CompositeJudge([])
        reward = composite.score(task_name="t", trajectory=_traj(), artifacts={})

        assert reward.factors == {}
        assert reward.scalarization == "weighted_log_product"
        assert reward.info["factor_weights"] == {}
