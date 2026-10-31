"""SkyRL integration layer.

Native integration with SkyRL's ecosystem:
- loss.py: GFN loss functions registered via register_policy_loss
- trainer.py: adapts the SkyRL trainer to GFlowNet losses; the policy worker
  manages log Z
- runner.py: Thin CLI entrypoint (runner=skyrl)
- registry.py: Centralized SkyRL registration
"""

from synthstats.integrations.skyrl.registry import (
    SKYRL_AVAILABLE,
    register_all,
    sync_registries_if_ray,
)

__all__ = [
    "SKYRL_AVAILABLE",
    "register_all",
    "sync_registries_if_ray",
]
