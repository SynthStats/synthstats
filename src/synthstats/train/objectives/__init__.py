"""Objectives - loss computation."""

from synthstats.train.objectives.catalog import (
    ADVANTAGE_ESTIMATORS,
    GFN_POLICY_LOSSES,
    TB_POLICY_LOSSES,
)
from synthstats.train.objectives.gfn_losses import GFNLossConfig, gfn_subtb_loss, gfn_tb_loss
from synthstats.train.objectives.registration import register_named_callables

__all__ = [
    "ADVANTAGE_ESTIMATORS",
    "GFN_POLICY_LOSSES",
    "GFNLossConfig",
    "TB_POLICY_LOSSES",
    "gfn_subtb_loss",
    "gfn_tb_loss",
    "register_named_callables",
]
