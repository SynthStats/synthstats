"""Registration helpers for objective catalogs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

NamedCallableCatalog = Mapping[str, Callable[..., Any]]
DecoratorRegistrar = Callable[[str], Callable[[Callable[..., Any]], Any]]


def register_named_callables(
    register: DecoratorRegistrar,
    catalog: NamedCallableCatalog,
) -> tuple[str, ...]:
    """Apply a decorator-style registry function to every named callable."""
    registered: list[str] = []
    for name, fn in catalog.items():
        try:
            register(name)(fn)
        except ValueError as exc:
            if "already registered" not in str(exc):
                raise
        registered.append(name)
    return tuple(registered)


__all__ = ["register_named_callables"]
