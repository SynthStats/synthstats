"""Exercise submitted programs in the actual Docker execution image."""

import json
import os
import subprocess
from pathlib import Path

import pytest


def test_program_cannot_write_the_controllers_mounted_directory(tmp_path):
    image = os.environ.get("SYNTHSTATS_TEST_IMAGE", "").strip()
    if not image:
        pytest.skip("SYNTHSTATS_TEST_IMAGE is not set")
    source = Path(__file__).resolve().parents[1] / "src"
    controller = """
import json
from pathlib import Path
from synthstats.executors.isolated_python import run_isolated_python
Path('/logs/controller.txt').write_text('reachable')
blocked = run_isolated_python(
    "from numpy import savetxt as save; save('/logs/escape.csv', [1])", timeout=15)
valid = run_isolated_python("import numpy as np; print(np.arange(3).sum())", timeout=15)
print(json.dumps({'blocked': blocked.returncode, 'valid': valid.returncode,
                  'stdout': valid.stdout, 'stderr': valid.stderr}))
"""
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--cap-add",
            "SYS_ADMIN",
            "--security-opt",
            "seccomp=unconfined",
            "--security-opt",
            "apparmor=unconfined",
            "--security-opt",
            "systempaths=unconfined",
            "--mount",
            f"type=bind,src={source},dst=/code,readonly",
            "--mount",
            f"type=bind,src={tmp_path},dst=/logs",
            "--env",
            "PYTHONPATH=/code",
            image,
            "python",
            "-c",
            controller,
        ],
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    payload = json.loads(result.stdout)
    assert (tmp_path / "controller.txt").read_text() == "reachable"
    assert payload["blocked"] != 0
    assert not (tmp_path / "escape.csv").exists()
    assert payload["valid"] == 0, payload["stderr"]
    assert payload["stdout"].strip() == "3"
