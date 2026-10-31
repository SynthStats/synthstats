"""Shared helpers for accessing values from dict-or-object containers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def config_get(container: Any, key: str, default: Any = None) -> Any:
    """Get a value from a dict, Mapping, or object with attribute access."""
    if container is None:
        return default
    if isinstance(container, Mapping):
        return container.get(key, default)
    if hasattr(container, key):
        return getattr(container, key)
    if hasattr(container, "get"):
        return container.get(key, default)
    return default
