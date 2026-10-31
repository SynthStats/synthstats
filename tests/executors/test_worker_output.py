"""Library diagnostics larger than a pipe cannot block a request."""

from synthstats.executors.worker_pool import PyMCWorkerPool


def test_large_pymc_warning_and_following_request_complete():
    with PyMCWorkerPool(pool_size=1) as pool:
        result = pool.submit(
            "import pymc; import logging; "
            "logging.getLogger('pymc').warning('x' * 100000); print(42)",
            timeout=30.0,
        )
        following = pool.submit("print(43)", timeout=30.0)
    assert result["returncode"] == 0, result.get("stderr")
    assert result["stdout"].strip() == "42"
    assert len(result["stderr"]) >= 100000
    assert following["returncode"] == 0
    assert following["stdout"].strip() == "43"
