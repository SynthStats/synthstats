"""Objective catalogs for training integrations."""

from __future__ import annotations

from synthstats.train.objectives.gfn_losses import gfn_subtb_loss, gfn_tb_loss
from synthstats.train.objectives.trajectory_balance import (
    compute_modified_subtb_loss,
    compute_tb_identity_advantage,
    compute_tb_log_reward_advantage,
    compute_trajectory_balance_loss,
)

GFN_POLICY_LOSSES = {
    "gfn_tb": gfn_tb_loss,
    "gfn_subtb": gfn_subtb_loss,
}

TB_POLICY_LOSSES = {
    "trajectory_balance": compute_trajectory_balance_loss,
    "modified_subtb": compute_modified_subtb_loss,
}

ADVANTAGE_ESTIMATORS = {
    "tb_identity": compute_tb_identity_advantage,
    "tb_log_reward": compute_tb_log_reward_advantage,
}

__all__ = [
    "ADVANTAGE_ESTIMATORS",
    "GFN_POLICY_LOSSES",
    "TB_POLICY_LOSSES",
]
