"""Execute Hydra application dispatch, including its reserved configuration."""

import subprocess
import sys
from pathlib import Path


def test_hydra_dispatch_reaches_runner(tmp_path):
    configs = Path(__file__).resolve().parents[2] / "src" / "synthstats" / "configs"
    script = """
from types import SimpleNamespace
from synthstats.cli import train
from synthstats.train.runners.base import RunResult

def run():
    print('RUNNER_DISPATCHED')
    return RunResult(metrics={'loss': 3.5})

train.build_runner = lambda cfg: SimpleNamespace(run=run)
train.main()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, "--config-path", str(configs), f"hydra.run.dir={tmp_path}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RUNNER_DISPATCHED" in result.stdout


def test_training_failure_exits_nonzero(tmp_path):
    configs = Path(__file__).resolve().parents[2] / "src" / "synthstats" / "configs"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "synthstats.cli.train",
            "--config-path",
            str(configs),
            "objective.loss_type=invalid",
            f"hydra.run.dir={tmp_path}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "not supported by the training CLI" in result.stdout + result.stderr
