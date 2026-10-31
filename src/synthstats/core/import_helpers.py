"""Generic import helpers for optional dependencies."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def try_import_symbol(symbol: str, module: str) -> Any | None:
    """Read an optional SDK symbol and expose broken installed dependencies."""
    try:
        mod = __import__(module, fromlist=[symbol])
        return getattr(mod, symbol)
    except ModuleNotFoundError as exc:
        if exc.name and (module == exc.name or module.startswith(exc.name + ".")):
            return None
        raise
    except AttributeError:
        logger.debug("Could not import %s from %s", symbol, module)
        return None
