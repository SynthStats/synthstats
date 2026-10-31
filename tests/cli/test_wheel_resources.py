"""Installed distributions include the configuration and task resources."""

import subprocess
import zipfile
from pathlib import Path


def test_wheel_contains_runtime_resources(tmp_path):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    (wheel,) = tmp_path.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        paths = set(archive.namelist())
    assert "synthstats/configs/train.yaml" in paths
    assert "synthstats/configs/objective/tb.yaml" in paths
    assert (
        "synthstats/integrations/harbor/task_packages/dugongs/environment/docker-compose.yaml"
        in paths
    )
