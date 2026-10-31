"""Persistent worker pool for PyMC program evaluation.

Persistent supervisors accept JSON requests on standard input and return JSON
responses on standard output. Each request runs in a fresh isolated interpreter.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from synthstats.core.constants import COMPILATION_TIMEOUT_S, PYMC_EXECUTOR_TIMEOUT_S
from synthstats.core.process import cleanup_process_group
from synthstats.executors._pymc_worker import END_REQUEST, END_RESPONSE, READY_SENTINEL

logger = logging.getLogger(__name__)

_WORKER_SCRIPT = str(Path(__file__).parent / "_pymc_worker.py")


class WorkerError(RuntimeError):
    """Raised when a worker fails to start or dies unexpectedly."""


class _Worker:
    """A single persistent worker subprocess."""

    def __init__(self, worker_id: int) -> None:
        self.worker_id = worker_id
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def start(self, timeout: float = COMPILATION_TIMEOUT_S) -> None:
        """Start the worker process and wait for the ready sentinel."""
        self.proc = subprocess.Popen(
            [sys.executable, _WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        # wait for the ready sentinel (with timeout via a thread)
        ready_line: str | None = None
        error_msg: str = ""

        def _read_ready() -> None:
            nonlocal ready_line, error_msg
            try:
                assert self.proc is not None
                assert self.proc.stdout is not None
                ready_line = self.proc.stdout.readline().strip()
            except Exception as exc:
                error_msg = str(exc)

        reader = threading.Thread(target=_read_ready, daemon=True)
        reader.start()
        reader.join(timeout=timeout)

        if reader.is_alive():
            self.shutdown()
            raise WorkerError(f"Worker {self.worker_id} did not become ready within {timeout}s")

        if ready_line != READY_SENTINEL:
            stderr_snippet = ""
            if self.proc and self.proc.stderr:
                try:
                    stderr_snippet = self.proc.stderr.read(500)
                except Exception:  # noqa: BLE001  # best-effort read; ignore all errors
                    pass
            self.shutdown()
            raise WorkerError(
                f"Worker {self.worker_id} sent unexpected ready line: "
                f"{ready_line!r}; stderr: {stderr_snippet!r}; {error_msg}"
            )

    def submit(
        self,
        code: str,
        seed: int | None = None,
        timeout: float = PYMC_EXECUTOR_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Send code to the worker, return the result dict.

        Note: holds the worker lock for the full request duration (up to timeout).
        Concurrent submits to the same worker serialize; the pool's round-robin
        distributes across workers to minimize contention.
        """
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                raise WorkerError(f"Worker {self.worker_id} is not running")

            assert self.proc.stdin is not None
            assert self.proc.stdout is not None

            request = json.dumps({"code": code, "seed": seed, "timeout": timeout})
            self.proc.stdin.write(request + "\n")
            self.proc.stdin.write(END_REQUEST + "\n")
            self.proc.stdin.flush()

            # read response with timeout
            result: dict[str, Any] | None = None
            read_error: str = ""

            def _read_response() -> None:
                nonlocal result, read_error
                try:
                    lines: list[str] = []
                    assert self.proc is not None
                    assert self.proc.stdout is not None
                    for raw in self.proc.stdout:
                        line = raw.rstrip("\n")
                        if line == END_RESPONSE:
                            break
                        lines.append(line)
                    result = json.loads("\n".join(lines))
                except Exception as exc:
                    read_error = str(exc)

            reader = threading.Thread(target=_read_response, daemon=True)
            reader.start()
            reader.join(timeout=timeout)

            if reader.is_alive():
                # timeout - kill the worker, caller should respawn
                self._force_kill()
                raise WorkerError(f"Worker {self.worker_id} timed out after {timeout}s")

            if result is None:
                self._force_kill()
                raise WorkerError(f"Worker {self.worker_id} returned no result: {read_error}")

            return result

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _force_kill(self) -> None:
        if self.proc is not None:
            cleanup_process_group(self.proc)
            self.proc = None

    def shutdown(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is not None:
            self.proc = None
            return
        # close stdin so the worker exits its read loop
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cleanup_process_group(self.proc)
        self.proc = None


class PyMCWorkerPool:
    """Pool of persistent PyMC worker subprocesses.

    Usage::

        with PyMCWorkerPool(pool_size=2) as pool:
            result = pool.submit("print('hello')")
            print(result["stdout"])  # "hello\\n"
    """

    def __init__(
        self,
        pool_size: int | None = None,
        startup_timeout: float = COMPILATION_TIMEOUT_S,
    ) -> None:
        self.pool_size = pool_size or min(os.cpu_count() or 4, 4)
        self._startup_timeout = startup_timeout
        self._workers: list[_Worker] = []
        self._round_robin = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        for i in range(self.pool_size):
            w = _Worker(i)
            w.start(timeout=self._startup_timeout)
            self._workers.append(w)
        logger.info("PyMCWorkerPool started with %d workers", self.pool_size)

    def submit(
        self,
        code: str,
        seed: int | None = None,
        timeout: float = PYMC_EXECUTOR_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Submit code to a worker for execution, with auto-respawn on crash."""
        with self._lock:
            if not self._workers:
                raise WorkerError("Pool has no workers (not started or already shut down)")
            worker = self._workers[self._round_robin % len(self._workers)]
            self._round_robin += 1

            # respawn inside the lock to prevent two threads from racing
            # on the same dead worker (TOCTOU on is_alive + start)
            if not worker.is_alive():
                logger.warning("Worker %d died, respawning", worker.worker_id)
                worker.start(timeout=self._startup_timeout)

        return worker.submit(code, seed=seed, timeout=timeout)

    def shutdown(self) -> None:
        for w in self._workers:
            w.shutdown()
        self._workers.clear()
        logger.info("PyMCWorkerPool shut down")

    @property
    def alive_count(self) -> int:
        return sum(1 for w in self._workers if w.is_alive())

    def __enter__(self) -> PyMCWorkerPool:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


__all__ = ["PyMCWorkerPool", "WorkerError"]
