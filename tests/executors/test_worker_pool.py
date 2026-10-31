"""Tests for the persistent PyMC worker pool."""

from __future__ import annotations

import os
import signal
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from synthstats.executors.worker_pool import PyMCWorkerPool, WorkerError, _Worker

# ---------------------------------------------------------------------------
# _Worker unit tests
# ---------------------------------------------------------------------------


class TestWorkerLifecycle:
    def test_worker_starts_and_stops(self):
        w = _Worker(0)
        w.start(timeout=30.0)
        assert w.is_alive()
        w.shutdown()
        assert not w.is_alive()

    def test_worker_submit_simple_print(self):
        w = _Worker(0)
        w.start(timeout=30.0)
        try:
            result = w.submit("print('hello')", timeout=10.0)
            assert result["returncode"] == 0
            assert result["stdout"].strip() == "hello"
            assert result["error"] is None
        finally:
            w.shutdown()

    def test_worker_submit_returns_error_on_exception(self):
        w = _Worker(0)
        w.start(timeout=30.0)
        try:
            result = w.submit("raise ValueError('boom')", timeout=10.0)
            assert result["returncode"] == 1
            assert "boom" in result["error"]
        finally:
            w.shutdown()

    def test_worker_submit_multiple_requests(self):
        """A single worker handles sequential requests."""
        w = _Worker(0)
        w.start(timeout=30.0)
        try:
            r1 = w.submit("print(1+1)", timeout=10.0)
            r2 = w.submit("print(2+2)", timeout=10.0)
            assert r1["stdout"].strip() == "2"
            assert r2["stdout"].strip() == "4"
        finally:
            w.shutdown()

    def test_worker_submit_raises_when_not_running(self):
        w = _Worker(0)
        with pytest.raises(WorkerError, match="not running"):
            w.submit("print('nope')")

    def test_worker_shutdown_idempotent(self):
        w = _Worker(0)
        w.start(timeout=30.0)
        w.shutdown()
        w.shutdown()  # second call should not raise
        assert not w.is_alive()


# ---------------------------------------------------------------------------
# PyMCWorkerPool tests
# ---------------------------------------------------------------------------


class TestPoolStartStop:
    def test_pool_starts_and_stops(self):
        pool = PyMCWorkerPool(pool_size=2, startup_timeout=30.0)
        pool.start()
        assert pool.alive_count == 2
        pool.shutdown()
        assert pool.alive_count == 0

    def test_pool_context_manager(self):
        with PyMCWorkerPool(pool_size=1, startup_timeout=30.0) as pool:
            assert pool.alive_count == 1
        assert pool.alive_count == 0


class TestPoolSubmit:
    def test_submit_simple_code(self):
        with PyMCWorkerPool(pool_size=1, startup_timeout=30.0) as pool:
            result = pool.submit("print('hello')", timeout=10.0)
            assert result["returncode"] == 0
            assert "hello" in result["stdout"]

    def test_submit_code_with_computation(self):
        code = textwrap.dedent("""\
            import math
            print(math.sqrt(144))
        """)
        with PyMCWorkerPool(pool_size=1, startup_timeout=30.0) as pool:
            result = pool.submit(code, timeout=10.0)
            assert result["returncode"] == 0
            assert "12.0" in result["stdout"]

    def test_submit_code_with_seed(self):
        code = textwrap.dedent("""\
            import numpy as np
            print(np.random.rand())
        """)
        with PyMCWorkerPool(pool_size=1, startup_timeout=30.0) as pool:
            r1 = pool.submit(code, seed=42, timeout=10.0)
            r2 = pool.submit(code, seed=42, timeout=10.0)
            assert r1["stdout"].strip() == r2["stdout"].strip()

    def test_submit_captures_stderr(self):
        code = textwrap.dedent("""\
            import sys
            print("err msg", file=sys.stderr)
            print("ok")
        """)
        with PyMCWorkerPool(pool_size=1, startup_timeout=30.0) as pool:
            result = pool.submit(code, timeout=10.0)
            assert result["returncode"] == 0
            assert "err msg" in result["stderr"]
            assert "ok" in result["stdout"]

    def test_submit_error_code_returns_nonzero(self):
        with PyMCWorkerPool(pool_size=1, startup_timeout=30.0) as pool:
            result = pool.submit("1/0", timeout=10.0)
            assert result["returncode"] == 1
            assert result["error"] is not None
            assert "ZeroDivisionError" in (result["stderr"] or "")

    def test_submit_to_empty_pool_raises(self):
        pool = PyMCWorkerPool(pool_size=1)
        with pytest.raises(WorkerError, match="no workers"):
            pool.submit("print('nope')")


class TestPoolConcurrency:
    def test_concurrent_submissions(self):
        """Submit 4 programs simultaneously across 2 workers."""
        programs = [f"print({i}*{i})" for i in range(4)]
        expected = [str(i * i) for i in range(4)]

        with PyMCWorkerPool(pool_size=2, startup_timeout=30.0) as pool:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(pool.submit, code, None, 10.0) for code in programs]
                results = [f.result() for f in futures]

        outputs = sorted(r["stdout"].strip() for r in results)
        assert outputs == sorted(expected)

    def test_round_robin_distribution(self):
        """Requests are distributed across workers."""
        with PyMCWorkerPool(pool_size=2, startup_timeout=30.0) as pool:
            # submit 4 tasks sequentially
            for i in range(4):
                result = pool.submit(f"print({i})", timeout=10.0)
                assert result["returncode"] == 0
            # round-robin counter should have advanced
            assert pool._round_robin == 4


class TestPoolRecovery:
    @pytest.mark.skipif(os.name != "posix", reason="kill signals not available on Windows")
    def test_worker_crash_recovery(self):
        """After a worker is killed, the pool respawns it on next submit."""
        with PyMCWorkerPool(pool_size=1, startup_timeout=30.0) as pool:
            # first submit works
            r1 = pool.submit("print('alive')", timeout=10.0)
            assert r1["returncode"] == 0

            # kill the worker process
            worker = pool._workers[0]
            assert worker.proc is not None
            os.kill(worker.proc.pid, signal.SIGKILL)
            # give OS time to reap
            time.sleep(0.2)
            assert not worker.is_alive()

            # next submit should auto-respawn the worker
            r2 = pool.submit("print('back')", timeout=30.0)
            assert r2["returncode"] == 0
            assert "back" in r2["stdout"]

    def test_timeout_handling(self):
        """Code that sleeps forever triggers a timeout."""
        with PyMCWorkerPool(pool_size=1, startup_timeout=30.0) as pool:
            with pytest.raises(WorkerError, match="timed out"):
                pool.submit("import time; time.sleep(999)", timeout=2.0)

            # pool should still be usable after timeout (respawn)
            result = pool.submit("print('recovered')", timeout=30.0)
            assert result["returncode"] == 0
            assert "recovered" in result["stdout"]

    def test_syntax_error_returns_error_result(self):
        with PyMCWorkerPool(pool_size=1, startup_timeout=30.0) as pool:
            result = pool.submit("def oops(", timeout=10.0)
            assert result["returncode"] == 1
            assert result["error"] is not None


class TestPoolOutputTransport:
    """Workers preserve submitted stdout independently of its contents."""

    def test_stdout_preserves_score_like_text(self):
        code = textwrap.dedent("""\
            elpd_value = -123.45
            print(f"__ELPD__:{elpd_value}")
        """)
        with PyMCWorkerPool(pool_size=1, startup_timeout=30.0) as pool:
            result = pool.submit(code, timeout=10.0)
            assert result["returncode"] == 0
            assert "__ELPD__:-123.45" in result["stdout"]

    def test_numpy_available(self):
        code = textwrap.dedent("""\
            import numpy as np
            x = np.array([1, 2, 3])
            print(x.mean())
        """)
        with PyMCWorkerPool(pool_size=1, startup_timeout=30.0) as pool:
            result = pool.submit(code, timeout=10.0)
            assert result["returncode"] == 0
            assert "2.0" in result["stdout"]
