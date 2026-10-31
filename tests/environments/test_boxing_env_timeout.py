import subprocess

import pytest

from synthstats.executors.isolated_python import run_isolated_python


def test_isolated_interpreter_times_out_after_starting_the_loop():
    code = "print('loop-started', flush=True)\nwhile True: pass\n"
    with pytest.raises(subprocess.TimeoutExpired) as failure:
        run_isolated_python(code, timeout=2.0)
    output = failure.value.stdout
    if isinstance(output, bytes):
        output = output.decode()
    assert output == "loop-started\n"
