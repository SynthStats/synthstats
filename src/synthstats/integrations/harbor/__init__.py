"""Harbor integration layer.

SynthStats uses Harbor in two ways:

- local Harbor task packages + the live Harbor generator path for rollout
  orchestration
- offline utilities for importing, exporting and analysing Harbor's Agent
  Trajectory Interchange Format (ATIF)

Install the SDK and training integration with
``uv sync --extra harbor --extra skyrl``.

- converter.py: ATIF <-> SynthStats type conversion (pure functions)
- collector.py: HarborCollector reads ATIF trajectory files for training
- generator.py: Harbor trial launcher for SkyRL-native rollouts
- task_registry.py: local Harbor task-package registry
- config.py: Typed config dataclasses
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# module-scope probe: import is cheap and the result is needed at import time
# (callers use HARBOR_SDK_AVAILABLE as a guard before doing any SDK work)
HARBOR_SDK_AVAILABLE = False
try:
    import harbor  # noqa: F401

    HARBOR_SDK_AVAILABLE = True
except ImportError:
    pass


def is_harbor_available() -> bool:
    """True always: offline ATIF reading needs no external deps."""
    return True


__all__ = [
    "HARBOR_SDK_AVAILABLE",
    "is_harbor_available",
]
