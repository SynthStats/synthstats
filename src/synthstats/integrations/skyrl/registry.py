"""SkyRL loss and estimator registration.

Registers GFlowNet losses from synthstats.integrations.skyrl.loss and the loss
and advantage-estimator catalogues in synthstats.train.objectives.catalog.
"""

from __future__ import annotations

import logging

from synthstats.core.import_helpers import try_import_symbol
from synthstats.envs.registration import register_environments
from synthstats.train.objectives.catalog import ADVANTAGE_ESTIMATORS, TB_POLICY_LOSSES
from synthstats.train.objectives.registration import register_named_callables

logger = logging.getLogger(__name__)

_REGISTERED = False

register_policy_loss = try_import_symbol(
    "register_policy_loss",
    "skyrl.backends.skyrl_train.utils.ppo_utils",
)
SKYRL_AVAILABLE = register_policy_loss is not None


def register_all() -> None:
    """Register all GFN losses, estimators, and environments.

    Idempotent: repeated calls are no-ops after the first success.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    if not SKYRL_AVAILABLE:
        logger.warning("SkyRL not available, skipping loss registration")
        return

    from synthstats.integrations.skyrl.loss import register_gfn_losses

    register_gfn_losses()
    _register_tb_policy_losses()
    _register_advantage_estimators()
    register_environments()
    _REGISTERED = True


def _register_advantage_estimators() -> None:
    register_advantage_estimator = try_import_symbol(
        "register_advantage_estimator",
        "skyrl.backends.skyrl_train.utils.ppo_utils",
    )
    if register_advantage_estimator is None:
        logger.debug("Could not register advantage estimators: SkyRL not available")
        return

    for name in register_named_callables(
        register_advantage_estimator,
        ADVANTAGE_ESTIMATORS,
    ):
        logger.debug("registered %s estimator", name)


def _register_tb_policy_losses() -> None:
    if register_policy_loss is None:
        return

    register_named_callables(register_policy_loss, TB_POLICY_LOSSES)


def _import_sync_registries():
    """Import SkyRL's sync_registries (pushes local registry to Ray NamedActor)."""
    return try_import_symbol(
        "sync_registries",
        "skyrl.backends.skyrl_train.utils.ppo_utils",
    )


def sync_registries_if_ray() -> None:
    """Sync loss registries across Ray workers (call after ray.init)."""
    if not SKYRL_AVAILABLE:
        return

    try:
        import ray

        if not ray.is_initialized():
            logger.warning("Ray not initialized, skipping registry sync")
            return

        register_all()

        skyrl_sync = _import_sync_registries()
        if skyrl_sync is not None:
            skyrl_sync()
            logger.debug("registry synced to Ray")
        else:
            logger.debug("registrations stored locally")

    except ImportError:
        logger.warning("Ray not available, skipping sync")
    except (RuntimeError, OSError, ValueError, TypeError) as e:
        logger.warning("Registry sync failed: %s", e)
