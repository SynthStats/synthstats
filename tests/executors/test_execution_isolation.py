"""Candidate requests cannot alter host files or subsequent execution state."""

from synthstats.core.types import ToolCall
from synthstats.executors.pymc_sandbox import PyMCExecutor, check_code_safety
from synthstats.executors.worker_pool import PyMCWorkerPool


def test_failed_request_cannot_mutate_next_request_modules():
    with PyMCWorkerPool(pool_size=1) as pool:
        before = pool.submit("import numpy as np; print(np.pi)")
        failed = pool.submit("import numpy as np; np.pi = 0; raise RuntimeError('failed')")
        after = pool.submit("import numpy as np; print(np.pi)")
    assert before["returncode"] == 0
    assert failed["returncode"] != 0
    assert after["returncode"] == 0
    assert before["stdout"] == after["stdout"] == "3.141592653589793\n"


def test_worker_cannot_write_outside_request_filesystem(tmp_path):
    marker = tmp_path / "outside.txt"
    with PyMCWorkerPool(pool_size=1) as pool:
        result = pool.submit(f"from numpy import savetxt as write; write({str(marker)!r}, [1])")
        control = pool.submit("import numpy as np; np.savetxt('/tmp/inside.txt', [42]); print(42)")
    assert not marker.exists()
    assert result["returncode"] != 0
    assert control["returncode"] == 0
    assert control["stdout"].strip() == "42"


def test_pymc_executor_cannot_write_host_file(tmp_path):
    marker = tmp_path / "outside.txt"
    result = PyMCExecutor().execute(
        ToolCall(
            name="pymc",
            raw="",
            input={"code": f"from numpy import savetxt as write; write({str(marker)!r}, [1])"},
        ),
        timeout_s=30,
    )
    assert not marker.exists()
    assert not result.success


def test_pymc_executor_runs_real_python():
    result = PyMCExecutor().execute(
        ToolCall(
            name="pymc", raw="", input={"code": "import numpy as np; print(np.arange(3).sum())"}
        ),
        timeout_s=30,
    )
    assert result.success, result.error
    assert result.output.strip() == "3"


def test_import_alias_cannot_bypass_file_io_check():
    safe, reason = check_code_safety("from numpy import savetxt as write; write('/tmp/a', [1])")
    assert not safe
    assert reason
