"""Training config validation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def resolve_algorithm_config(config: Any) -> Any:
    """Return the effective algorithm config from nested trainer configs."""
    algorithm = getattr(config, "trainer", config)
    if hasattr(algorithm, "algorithm"):
        return algorithm.algorithm
    if isinstance(algorithm, Mapping):
        nested = algorithm.get("algorithm")
        if nested is not None:
            return nested
    return algorithm


def validate_gfn_algorithm_config(config: Any) -> None:
    """Reject algorithm settings that do not match SynthStats' GFN path."""
    algorithm = resolve_algorithm_config(config)

    loss_type = getattr(algorithm, "policy_loss_type", None)
    if isinstance(algorithm, Mapping):
        loss_type = algorithm.get("policy_loss_type", loss_type)
    if loss_type is None:
        loss_type = "regular"
    if loss_type not in ("gfn_tb", "gfn_subtb"):
        raise ValueError(
            "GFlowNetTrainer requires trainer.algorithm.policy_loss_type to be "
            f"'gfn_tb' or 'gfn_subtb'; got {loss_type!r}"
        )

    if loss_type == "gfn_subtb":
        raise ValueError(
            "gfn_subtb requires aligned, differentiable current-policy EOS probabilities "
            "from the policy worker. This runner does not supply them; select gfn_tb."
        )

    adv_estimator = getattr(algorithm, "advantage_estimator", None)
    if isinstance(algorithm, Mapping):
        adv_estimator = algorithm.get("advantage_estimator", adv_estimator)
    if adv_estimator is None:
        adv_estimator = "gae"
    if adv_estimator != "tb_log_reward":
        raise ValueError(
            "GFlowNetTrainer requires trainer.algorithm.advantage_estimator to be "
            f"'tb_log_reward'; got {adv_estimator!r}"
        )

    batch_norm = getattr(algorithm, "advantage_batch_normalize", False)
    if isinstance(algorithm, Mapping):
        batch_norm = algorithm.get("advantage_batch_normalize", batch_norm)
    if bool(batch_norm):
        raise ValueError(
            "GFlowNetTrainer requires trainer.algorithm.advantage_batch_normalize=False; "
            "TB log-rewards must not be z-score normalized"
        )


__all__ = ["resolve_algorithm_config", "validate_gfn_algorithm_config"]
