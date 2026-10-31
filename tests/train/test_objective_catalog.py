from __future__ import annotations

from synthstats.train.objectives import (
    ADVANTAGE_ESTIMATORS,
    GFN_POLICY_LOSSES,
    TB_POLICY_LOSSES,
)
from synthstats.train.objectives.gfn_losses import gfn_subtb_loss, gfn_tb_loss
from synthstats.train.objectives.trajectory_balance import (
    compute_modified_subtb_loss,
    compute_tb_identity_advantage,
    compute_tb_log_reward_advantage,
    compute_trajectory_balance_loss,
)


class TestObjectiveCatalog:
    def test_gfn_policy_losses_use_gfn_module(self):
        assert GFN_POLICY_LOSSES == {
            "gfn_tb": gfn_tb_loss,
            "gfn_subtb": gfn_subtb_loss,
        }

    def test_tb_policy_losses_point_to_first_party_objectives(self):
        assert TB_POLICY_LOSSES == {
            "trajectory_balance": compute_trajectory_balance_loss,
            "modified_subtb": compute_modified_subtb_loss,
        }

    def test_advantage_estimators_catalog_exposes_tb_identity(self):
        assert ADVANTAGE_ESTIMATORS == {
            "tb_identity": compute_tb_identity_advantage,
            "tb_log_reward": compute_tb_log_reward_advantage,
        }
