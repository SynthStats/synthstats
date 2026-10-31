"""SkyRL registration shell for first-party GFlowNet loss implementations."""

from __future__ import annotations

import logging

from synthstats.train.objectives.catalog import GFN_POLICY_LOSSES
from synthstats.train.objectives.gfn_losses import GFNLossConfig, gfn_subtb_loss, gfn_tb_loss
from synthstats.train.objectives.registration import register_named_callables

logger = logging.getLogger(__name__)


def register_gfn_losses() -> bool:
    """Register GFlowNet losses with SkyRL. Returns True if successful."""
    from synthstats.integrations.skyrl.registry import register_policy_loss

    if register_policy_loss is None:
        logger.debug("SkyRL not available, skipping GFN loss registration")
        return False

    try:
        registered = register_named_callables(register_policy_loss, GFN_POLICY_LOSSES)
        logger.info("Registered %s", ", ".join(sorted(registered)))
        return True
    except (RuntimeError, ValueError, KeyError, TypeError) as e:
        logger.warning("GFN loss registration failed: %s", e)
        return False


__all__ = ["GFNLossConfig", "gfn_subtb_loss", "gfn_tb_loss", "register_gfn_losses"]
