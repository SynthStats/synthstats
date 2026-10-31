import math
from types import SimpleNamespace

import pytest
import torch

from synthstats.train.experiment_config import build_training_experiment_config
from synthstats.train.gfn_core import LogZModule
from synthstats.train.objectives.catalog import ADVANTAGE_ESTIMATORS
from synthstats.train.objectives.gfn_losses import gfn_tb_loss


def estimate(reward, *, floor=1e-4):
    objective = {"loss_type": "tb"}
    if floor is not None:
        objective["reward_floor"] = floor
    cfg = build_training_experiment_config(
        {
            "objective": objective,
            "runner": {"trainer": {}, "generator": {}},
        }
    )
    algorithm = cfg.trainer.algorithm
    estimator = ADVANTAGE_ESTIMATORS[algorithm.advantage_estimator]
    return estimator(
        torch.tensor([[reward]], dtype=torch.float64),
        torch.ones(1, 1),
        config=algorithm,
        index=["trajectory-0"],
    )[0]


def test_raw_reward_e_enters_tb_as_one():
    advantages = estimate(math.e)
    assert advantages.item() == pytest.approx(1.0)
    loss, _ = gfn_tb_loss(
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        advantages,
        SimpleNamespace(logZ_module=LogZModule(0.0)),
    )
    assert loss.item() == pytest.approx(1.0)


def test_zero_reward_uses_the_declared_floor():
    assert estimate(0.0).item() == pytest.approx(math.log(1e-4))


@pytest.mark.parametrize("reward", [-1.0, float("nan"), float("inf")])
def test_invalid_reward_names_the_trajectory(reward):
    with pytest.raises(ValueError, match="trajectory-0"):
        estimate(reward)


def test_raw_reward_estimation_requires_the_floor():
    with pytest.raises(ValueError, match="reward_floor"):
        estimate(1.0, floor=None)


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), -float("inf")])
def test_tb_rejects_nonfinite_log_rewards(reward):
    with pytest.raises(ValueError, match="log reward"):
        gfn_tb_loss(
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.tensor([[reward]]),
            SimpleNamespace(logZ_module=LogZModule(0.0)),
        )
