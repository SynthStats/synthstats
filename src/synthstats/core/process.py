"""Process group cleanup and subprocess resource limits."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from typing import Any

from synthstats.core.constants import SAFE_ENV_VARS

logger = logging.getLogger(__name__)


def safe_subprocess_env(
    extra: dict[str, str] | None = None,
    *,
    allowed: frozenset[str] = SAFE_ENV_VARS,
) -> dict[str, str]:
    """Build a sanitized env dict from *allowed* vars, with optional extras."""
    env = {k: os.environ[k] for k in allowed if k in os.environ}
    if extra:
        env.update(extra)
    return env


def set_subprocess_rlimits(timeout_s: float, mem_gb: int = 2) -> None:
    """Set CPU and memory rlimits for a subprocess (POSIX only).

    Intended as a ``preexec_fn`` argument to ``subprocess.Popen``.
    Raises ``RuntimeError`` if the limits cannot be applied (so the caller
    sees a clean failure rather than a silent no-op).
    """
    try:
        import resource
    except ImportError as exc:
        logger.debug("resource module not available on this platform: %s", exc)
        return

    try:
        cpu_limit = max(1, int(timeout_s))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        if sys.platform == "linux":
            mem_limit = mem_gb * 1024 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
    except (ValueError, OSError) as exc:
        raise RuntimeError(f"failed to set subprocess resource limits: {exc}") from exc


def _group_exited(group_id: int) -> bool:
    try:
        os.killpg(group_id, 0)
        return False
    except (ProcessLookupError, OSError):
        return True


def _wait_group_exit(group_id: int, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _group_exited(group_id):
            return True
        time.sleep(0.05)
    return _group_exited(group_id)


def cleanup_process_group(proc: Any) -> None:
    """Best-effort SIGTERM -> SIGKILL on the process group, then reap."""
    pgid: int | None = None
    if hasattr(os, "killpg") and hasattr(os, "getpgid"):
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError) as exc:
            logger.debug("Could not resolve process group for pid %s: %s", proc.pid, exc)

    if pgid is not None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, OSError):
                break
            if _wait_group_exit(pgid):
                break

    try:
        proc.kill()
    except OSError as exc:
        logger.debug("Parent process kill failed during cleanup: %s", exc)
    try:
        proc.wait(timeout=2.0)
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("Parent process wait failed during cleanup: %s", exc)
