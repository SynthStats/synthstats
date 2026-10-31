"""Tests for LazyPPL log-likelihood pipeline."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from synthstats.envs.task_runtime_env import TaskRuntimeEnv, TaskRuntimeEnvConfig
from synthstats.executors.elpd import ELPDRunner
from synthstats.executors.lazyppl_sandbox import (
    LazyPPLExecutor,
    _format_hs_value,
    generate_taskdata_hs,
)
from synthstats.judges.likelihood import LikelihoodJudge


def _has_arviz() -> bool:
    try:
        import arviz  # noqa: F401

        return True
    except ImportError:
        return False


def _make_boxing_env(
    *,
    task_name: str = "dummy",
    task_env: object | None = None,
    judge: LikelihoodJudge | None = None,
    executors: dict[str, object] | None = None,
    config: TaskRuntimeEnvConfig | None = None,
) -> TaskRuntimeEnv:
    return TaskRuntimeEnv(
        task=SimpleNamespace(name=task_name, env=task_env),
        codec=SimpleNamespace(),
        executors=executors or {},
        judge=judge,
        config=config or TaskRuntimeEnvConfig(reward_floor=1e-4, reward_scale=1.0),
    )


# ---------------------------------------------------------------------------
# generate_taskdata_hs
# ---------------------------------------------------------------------------


class TestGenerateTaskdataHs:
    def test_basic_double_columns(self):
        columns = [
            {"name": "xVal", "hs_type": "Double"},
            {"name": "yVal", "hs_type": "Double"},
        ]
        rows = [
            {"xVal": 1.0, "yVal": 2.5},
            {"xVal": 3.0, "yVal": 1.8},
        ]
        hs = generate_taskdata_hs(columns, rows)
        assert "module TaskData" in hs
        assert "data Row = Row" in hs
        assert "xVal :: Double" in hs
        assert "yVal :: Double" in hs
        assert "loadData :: IO [Row]" in hs
        assert "1.0" in hs
        assert "2.5" in hs

    def test_mixed_types(self):
        columns = [
            {"name": "year", "hs_type": "Int"},
            {"name": "count", "hs_type": "Int"},
        ]
        rows = [{"year": 5, "count": 120}]
        hs = generate_taskdata_hs(columns, rows)
        assert "year :: Int" in hs
        assert "year = 5" in hs
        assert "count = 120" in hs

    def test_empty_rows(self):
        columns = [{"name": "x", "hs_type": "Double"}]
        hs = generate_taskdata_hs(columns, [])
        assert "loadData = return []" in hs

    def test_empty_columns_raises(self):
        with pytest.raises(ValueError, match="columns must not be empty"):
            generate_taskdata_hs([], [{"x": 1}])


class TestFormatHsValue:
    def test_double(self):
        assert _format_hs_value(3.14, "Double") == "3.14"

    def test_int(self):
        assert _format_hs_value(42, "Int") == "42"
        assert _format_hs_value(3.9, "Int") == "3"

    def test_bool(self):
        assert _format_hs_value(True, "Bool") == "True"
        assert _format_hs_value(False, "Bool") == "False"

    def test_nan(self):
        assert _format_hs_value(float("nan"), "Double") == "(0/0)"

    def test_inf(self):
        assert _format_hs_value(float("inf"), "Double") == "(1/0)"
        assert _format_hs_value(float("-inf"), "Double") == "(-(1/0))"


# ---------------------------------------------------------------------------
# Env eval data
# ---------------------------------------------------------------------------


class TestEnvEvalData:
    def test_dugongs_eval_data(self):
        from synthstats.tasks.statistical_modeling.envs.dugongs_env import DugongsEnv

        env = DugongsEnv()
        cols = env.get_lazyppl_columns()
        rows = env.get_lazyppl_eval_data()

        assert len(cols) == 2
        assert cols[0]["name"] == "xVal"
        assert cols[1]["name"] == "yVal"
        assert len(rows) == 10
        assert all("xVal" in r and "yVal" in r for r in rows)
        # ages should be in the expected range
        ages = [r["xVal"] for r in rows]
        assert min(ages) == 1.0
        assert max(ages) == 30.0

    def test_dugongs_eval_data_deterministic(self):
        from synthstats.tasks.statistical_modeling.envs.dugongs_env import DugongsEnv

        env = DugongsEnv()
        rows1 = env.get_lazyppl_eval_data()
        rows2 = env.get_lazyppl_eval_data()
        assert rows1 == rows2

    def test_peregrines_eval_data(self):
        from synthstats.tasks.statistical_modeling.envs.peregrines_env import PeregrinesEnv

        env = PeregrinesEnv()
        cols = env.get_lazyppl_columns()
        rows = env.get_lazyppl_eval_data()

        assert len(cols) == 2
        assert cols[0]["name"] == "year"
        assert cols[0]["hs_type"] == "Int"
        assert len(rows) == 10
        assert all(r["population"] >= 0 for r in rows)

    def test_eight_schools_eval_data(self):
        from synthstats.tasks.statistical_modeling.envs.eight_schools_env import EightSchoolsEnv

        env = EightSchoolsEnv()
        cols = env.get_lazyppl_columns()
        rows = env.get_lazyppl_eval_data()

        assert len(cols) == 2
        assert cols[0]["name"] == "effect"
        assert len(rows) == 8
        # known SEs from Rubin (1981)
        ses = [r["se"] for r in rows]
        assert ses == [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]

    def test_surgical_eval_data(self):
        from synthstats.tasks.statistical_modeling.envs.surgical_env import SurgicalEnv

        env = SurgicalEnv()
        cols = env.get_lazyppl_columns()
        rows = env.get_lazyppl_eval_data()

        assert len(cols) == 2
        assert cols[0]["name"] == "deaths"
        assert cols[1]["name"] == "exposure"
        assert len(rows) == 12
        assert all(r["deaths"] >= 0 and r["exposure"] > 0 for r in rows)

    def test_taskdata_hs_roundtrip_dugongs(self):
        """generate_taskdata_hs produces valid Haskell from dugongs data."""
        from synthstats.tasks.statistical_modeling.envs.dugongs_env import DugongsEnv

        env = DugongsEnv()
        cols = env.get_lazyppl_columns()
        rows = env.get_lazyppl_eval_data()
        hs = generate_taskdata_hs(cols, rows)

        assert "module TaskData" in hs
        assert "xVal :: Double" in hs
        assert "yVal :: Double" in hs
        # should have 10 Row entries
        assert hs.count("Row {") == 10


# ---------------------------------------------------------------------------
# Language propagation in TaskRuntimeEnv
# ---------------------------------------------------------------------------


class TestLanguagePropagation:
    def _make_env(self, judge=None, executors=None, task_env=None) -> TaskRuntimeEnv:
        return _make_boxing_env(judge=judge, executors=executors, task_env=task_env)

    def test_score_program_pymc_default(self, monkeypatch):
        """score_program with default language still works."""
        env = self._make_env(judge=LikelihoodJudge(beta=0.5))
        monkeypatch.setattr(
            ELPDRunner,
            "execute",
            lambda self, code: {
                "elpd": 10.0,
                "diagnostics": {"warning": False, "good_k": 0.7, "n_over_good_k": 0},
            },
        )
        reward = env.score_program("x = 1\n")
        assert reward == pytest.approx(math.exp(5.0))

    def test_score_program_lazyppl_language_in_artifacts(self, monkeypatch):
        """score_program with language='lazyppl' includes language in artifacts."""
        mock_executor = MagicMock()
        mock_executor.execute_for_loglik.return_value = None
        mock_env = SimpleNamespace(
            get_lazyppl_columns=lambda: [{"name": "x", "hs_type": "Double"}],
            get_lazyppl_eval_data=lambda: [{"x": 1.0}],
        )

        env = self._make_env(
            judge=LikelihoodJudge(beta=0.1),
            executors={"lazyppl": mock_executor},
            task_env=mock_env,
        )

        # should not call ELPDRunner.execute
        elpd_called = []
        monkeypatch.setattr(
            ELPDRunner,
            "execute",
            lambda self, code: elpd_called.append(True) or {"error": "should not be called"},
        )

        env.score_program("module Foo where", language="lazyppl")
        assert len(elpd_called) == 0

    def test_compute_final_reward_dispatches_lazyppl(self, monkeypatch):
        """_compute_final_reward calls _populate_lazyppl_loglik for lazyppl."""
        env = self._make_env(judge=LikelihoodJudge(beta=0.1))

        lazyppl_called = []
        monkeypatch.setattr(
            TaskRuntimeEnv,
            "_populate_lazyppl_loglik",
            lambda self, prog, arts: lazyppl_called.append(prog),
        )

        env.score_program("module Foo where", language="lazyppl")
        assert len(lazyppl_called) == 1
        assert lazyppl_called[0] == "module Foo where"

    def test_ensure_artifacts_computed_normalizes_haskell_alias(self, monkeypatch):
        env = self._make_env(judge=LikelihoodJudge(beta=0.1))
        lazyppl_called = []

        monkeypatch.setattr(
            TaskRuntimeEnv,
            "_populate_lazyppl_loglik",
            lambda self, prog, arts: lazyppl_called.append(prog),
        )

        env._ensure_artifacts_computed({"program": "module Foo where", "language": "haskell"})

        assert lazyppl_called == ["module Foo where"]

    def test_compute_final_reward_dispatches_pymc(self, monkeypatch):
        """_compute_final_reward calls ELPDRunner.execute for pymc."""
        env = self._make_env(judge=LikelihoodJudge(beta=0.1))

        pymc_called = []
        monkeypatch.setattr(
            ELPDRunner,
            "execute",
            lambda self, code: pymc_called.append(True) or {"error": "mock"},
        )

        env.score_program("x = 1", language="pymc")
        assert len(pymc_called) == 1

    def test_compute_final_reward_skips_pymc_for_lazyppl(self, monkeypatch):
        """_compute_final_reward does NOT call ELPDRunner.execute for lazyppl."""
        env = self._make_env(judge=LikelihoodJudge(beta=0.1))

        pymc_called = []
        monkeypatch.setattr(
            ELPDRunner,
            "execute",
            lambda self, code: pymc_called.append(True) or {"error": "should not be called"},
        )
        monkeypatch.setattr(
            TaskRuntimeEnv,
            "_populate_lazyppl_loglik",
            lambda self, prog, arts: None,
        )

        env.score_program("module Foo where", language="lazyppl")
        assert len(pymc_called) == 0


# ---------------------------------------------------------------------------
# _populate_lazyppl_loglik wiring
# ---------------------------------------------------------------------------


class TestExecuteLazypplForLoglik:
    def _make_env_with_executor(self, executor, env_obj=None):
        return _make_boxing_env(
            task_name="test",
            task_env=env_obj,
            judge=LikelihoodJudge(beta=0.1),
            executors={"lazyppl": executor},
        )

    def test_stores_log_lik_matrix_in_artifacts(self):
        fake_matrix = np.random.randn(200, 10)
        mock_executor = MagicMock()
        mock_executor.execute_for_loglik.return_value = fake_matrix
        mock_env = SimpleNamespace(
            get_lazyppl_columns=lambda: [{"name": "x", "hs_type": "Double"}],
            get_lazyppl_eval_data=lambda: [{"x": float(i)} for i in range(10)],
        )

        env = self._make_env_with_executor(mock_executor, mock_env)
        artifacts = {"program": "code", "language": "lazyppl"}
        env._populate_lazyppl_loglik("module Foo where", artifacts)

        assert "log_lik_matrix" in artifacts
        np.testing.assert_array_equal(artifacts["log_lik_matrix"], fake_matrix)

    def test_no_executor_logs_warning(self, caplog):
        boxing_env = _make_boxing_env(task_name="test")
        artifacts: dict[str, object] = {}

        with caplog.at_level("WARNING"):
            boxing_env._populate_lazyppl_loglik("code", artifacts)

        assert any("no lazyppl executor" in r.message for r in caplog.records)
        assert "log_lik_matrix" not in artifacts

    def test_no_eval_data_logs_warning(self, caplog):
        mock_executor = MagicMock()
        mock_executor.execute_for_loglik.return_value = None
        mock_env = SimpleNamespace()  # no get_lazyppl_columns

        env = self._make_env_with_executor(mock_executor, mock_env)
        artifacts = {"program": "code", "language": "lazyppl"}
        with caplog.at_level("WARNING"):
            env._populate_lazyppl_loglik("code", artifacts)

        assert any("does not provide lazyppl eval data" in r.message for r in caplog.records)

    def test_executor_returns_none_no_crash(self):
        mock_executor = MagicMock()
        mock_executor.execute_for_loglik.return_value = None
        mock_env = SimpleNamespace(
            get_lazyppl_columns=lambda: [{"name": "x", "hs_type": "Double"}],
            get_lazyppl_eval_data=lambda: [{"x": 1.0}],
        )

        env = self._make_env_with_executor(mock_executor, mock_env)
        artifacts = {"program": "code", "language": "lazyppl"}
        env._populate_lazyppl_loglik("code", artifacts)
        assert "log_lik_matrix" not in artifacts

    def test_task_env_none_logs_warning(self, caplog):
        """task.env being None should log warning, not crash."""
        mock_executor = MagicMock()
        boxing_env = _make_boxing_env(
            task_name="test",
            task_env=None,
            executors={"lazyppl": mock_executor},
        )
        artifacts: dict[str, object] = {}

        with caplog.at_level("WARNING"):
            boxing_env._populate_lazyppl_loglik("code", artifacts)

        assert any("does not provide lazyppl eval data" in r.message for r in caplog.records)
        assert "log_lik_matrix" not in artifacts


# ---------------------------------------------------------------------------
# Integration: log_lik_matrix → LikelihoodJudge → reward
# ---------------------------------------------------------------------------


class TestLoglikToReward:
    @pytest.mark.skipif(not _has_arviz(), reason="arviz not installed")
    def test_log_lik_matrix_produces_reward(self):
        """LikelihoodJudge produces non-zero reward from a log_lik_matrix."""
        from synthstats.core.types import Reward, Trajectory

        judge = LikelihoodJudge(beta=0.1)
        n_draws, n_obs = 200, 10
        # log-lik values around -1 (reasonable)
        log_lik = np.random.RandomState(42).normal(-1.0, 0.5, (n_draws, n_obs))

        traj = Trajectory(
            messages=[],
            token_ids=[],
            token_logprobs=[],
            loss_mask=[],
            reward=Reward(0.0, {}, {}),
        )
        reward = judge.score(
            task_name="test",
            trajectory=traj,
            artifacts={"log_lik_matrix": log_lik},
        )

        assert reward.total > 0
        assert reward.info["elpd_source"] == "log_lik_matrix_loo"
        assert "elpd" in reward.components

    def test_full_pipeline_lazyppl_artifacts(self):
        """TaskRuntimeEnv with lazyppl artifacts and LikelihoodJudge produces reward."""
        n_draws, n_obs = 200, 10
        # Keep the mocked likelihood surface low-variance so strict PSIS-LOO accepts it
        # deterministically; high-variance toy matrices can trip the Pareto-k gate.
        log_lik = np.random.RandomState(0).normal(-1.0, 0.01, (n_draws, n_obs))
        mock_executor = MagicMock()
        mock_executor.execute_for_loglik.return_value = log_lik
        mock_env = SimpleNamespace(
            get_lazyppl_columns=lambda: [{"name": "x", "hs_type": "Double"}],
            get_lazyppl_eval_data=lambda: [{"x": float(i)} for i in range(n_obs)],
        )

        env = _make_boxing_env(
            task_name="test",
            task_env=mock_env,
            judge=LikelihoodJudge(beta=0.1),
            executors={"lazyppl": mock_executor},
        )

        reward = env.score_program("module Foo where", language="lazyppl")
        assert reward > env.config.reward_floor


# ---------------------------------------------------------------------------
# LazyPPLExecutor.execute_for_loglik unit tests (mocked subprocess)
# ---------------------------------------------------------------------------


class TestExecuteForLoglikUnit:
    def test_parses_valid_json_output(self, tmp_path, monkeypatch):
        """Mocked subprocess returning valid JSON → numpy array."""
        import json

        n_draws, n_obs = 5, 3
        log_lik = [[float(i + j) for j in range(n_obs)] for i in range(n_draws)]
        json_output = json.dumps({"log_lik": log_lik, "n_draws": n_draws, "n_obs": n_obs})

        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (json_output, "")
        mock_proc.returncode = 0

        monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: mock_proc)

        executor = LazyPPLExecutor(lazyppl_dir=str(tmp_path))
        result = executor.execute_for_loglik(
            "module Foo where",
            [{"name": "x", "hs_type": "Double"}],
            [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}],
        )

        assert result is not None
        assert result.shape == (n_draws, n_obs)
        np.testing.assert_array_equal(result[0], [0.0, 1.0, 2.0])

    def test_handles_error_json(self, tmp_path, monkeypatch):
        """Mocked subprocess returning error JSON → None."""
        import json

        json_output = json.dumps({"error": "something went wrong"})
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (json_output, "")
        mock_proc.returncode = 0

        monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: mock_proc)

        executor = LazyPPLExecutor(lazyppl_dir=str(tmp_path))
        result = executor.execute_for_loglik(
            "module Foo where",
            [{"name": "x", "hs_type": "Double"}],
            [{"x": 1.0}],
        )
        assert result is None

    def test_handles_nonzero_exit_code(self, tmp_path, monkeypatch):
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("", "compilation error")
        mock_proc.returncode = 1

        monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: mock_proc)

        executor = LazyPPLExecutor(lazyppl_dir=str(tmp_path))
        result = executor.execute_for_loglik(
            "module Foo where",
            [{"name": "x", "hs_type": "Double"}],
            [{"x": 1.0}],
        )
        assert result is None

    def test_module_renamed_to_submission(self, tmp_path, monkeypatch):
        """Verifies the submitted code is rewritten to module Submission."""
        captured_files = {}

        original_open = open

        def mock_open_capture(path, mode="r", *a, **kw):
            f = original_open(path, mode, *a, **kw)
            if mode == "w" and "Submission.hs" in str(path):
                original_write = f.write

                def capture_write(data):
                    captured_files["Submission.hs"] = data
                    return original_write(data)

                f.write = capture_write
            return f

        monkeypatch.setattr("builtins.open", mock_open_capture)

        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ('{"log_lik":[[1.0]],"n_draws":1,"n_obs":1}', "")
        mock_proc.returncode = 0
        monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: mock_proc)

        executor = LazyPPLExecutor(lazyppl_dir=str(tmp_path))
        executor.execute_for_loglik(
            "module MyModule where\nimport LazyPPL",
            [{"name": "x", "hs_type": "Double"}],
            [{"x": 1.0}],
        )

        assert "Submission.hs" in captured_files
        assert "module Submission" in captured_files["Submission.hs"]
        assert "module MyModule" not in captured_files["Submission.hs"]

    def test_timeout_returns_none(self, tmp_path, monkeypatch):
        """Subprocess timeout → None."""
        import subprocess

        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = subprocess.TimeoutExpired(cmd="runhaskell", timeout=1)
        monkeypatch.setattr("os.getpgid", MagicMock(side_effect=ProcessLookupError))

        monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: mock_proc)

        executor = LazyPPLExecutor(lazyppl_dir=str(tmp_path))
        result = executor.execute_for_loglik(
            "module Foo where",
            [{"name": "x", "hs_type": "Double"}],
            [{"x": 1.0}],
        )
        assert result is None

    def test_malformed_json_returns_none(self, tmp_path, monkeypatch):
        """Non-JSON stdout → None."""
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("not valid json{{{", "")
        mock_proc.returncode = 0

        monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: mock_proc)

        executor = LazyPPLExecutor(lazyppl_dir=str(tmp_path))
        result = executor.execute_for_loglik(
            "module Foo where",
            [{"name": "x", "hs_type": "Double"}],
            [{"x": 1.0}],
        )
        assert result is None

    def test_json_with_preamble_still_parses(self, tmp_path, monkeypatch):
        """JSON output preceded by GHC noise still parses."""
        import json

        log_lik = [[1.0, 2.0], [3.0, 4.0]]
        json_line = json.dumps({"log_lik": log_lik, "n_draws": 2, "n_obs": 2})
        stdout = f"Resolving dependencies...\nSome GHC warning\n{json_line}\n"

        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (stdout, "")
        mock_proc.returncode = 0

        monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: mock_proc)

        executor = LazyPPLExecutor(lazyppl_dir=str(tmp_path))
        result = executor.execute_for_loglik(
            "module Foo where",
            [{"name": "x", "hs_type": "Double"}],
            [{"x": 1.0}, {"x": 2.0}],
        )

        assert result is not None
        assert result.shape == (2, 2)


class TestScoreProgramEdgeCases:
    def test_unknown_language_logs_warning(self, caplog):
        """score_program with unsupported language logs a warning."""
        env = _make_boxing_env(judge=LikelihoodJudge(beta=0.1))

        with caplog.at_level("WARNING"):
            reward = env.score_program("x = 1", language="numpyro")

        assert any("unsupported language" in r.message for r in caplog.records)
        assert reward >= env.config.reward_floor


class TestLazyPPLIsolationBoundary:
    def test_lazyppl_subprocesses_use_bubblewrap_isolation(self, tmp_path, monkeypatch):
        import os

        from synthstats.core.types import ToolCall

        captured_cmds: list[list[str]] = []

        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (
            'Ok, 1 modules loaded.\n{"log_lik":[[1.0,2.0],[1.5,2.5]],"n_draws":2,"n_obs":2}',
            "",
        )
        mock_proc.returncode = 0
        mock_proc.poll.return_value = 0

        def fake_popen(cmd, *args, **kwargs):
            captured_cmds.append(list(cmd))
            return mock_proc

        monkeypatch.setattr("subprocess.Popen", fake_popen)

        executor = LazyPPLExecutor(lazyppl_dir=str(tmp_path), compile_only=False)
        res = executor.execute(
            ToolCall(
                name="lazyppl", input={"code": 'module M where\nmain = putStrLn "ok"'}, raw=""
            ),
            timeout_s=5.0,
        )
        assert res.success is True
        res_loglik = executor.execute_for_loglik(
            "module Submission where\nmodel rows = return ()",
            [{"name": "x", "hs_type": "Double"}],
            [{"x": 1.0}, {"x": 2.0}],
        )
        assert res_loglik is not None

        assert len(captured_cmds) == 3
        for cmd in captured_cmds:
            assert os.path.basename(cmd[0]) == "bwrap"
            assert "--unshare-all" in cmd
            assert "--remount-ro" in cmd
            tmpfs_idx = next(
                i for i in range(len(cmd) - 1) if cmd[i] == "--tmpfs" and cmd[i + 1] == "/tmp"
            )
            tmp_ro_binds = [
                i
                for i in range(len(cmd) - 2)
                if cmd[i] == "--ro-bind" and cmd[i + 1].startswith("/tmp/")
            ]
            assert tmp_ro_binds
            assert all(tmpfs_idx < bind_idx for bind_idx in tmp_ro_binds)

    def test_lazyppl_real_bubblewrap_exposes_generated_tmp_inputs(self, monkeypatch):
        import os
        import shutil
        import sys
        import tempfile
        from pathlib import Path

        from synthstats.core.types import ToolCall

        if sys.platform != "linux" or shutil.which("bwrap") is None:
            pytest.skip("Linux bubblewrap required")

        host_canary = tempfile.NamedTemporaryFile(prefix="lazyppl_host_canary_", delete=False)
        host_canary.write(b"canary")
        host_canary.close()

        checkout_parent = "/var/tmp" if os.path.isdir("/var/tmp") else None
        checkout_dir = Path(
            tempfile.mkdtemp(prefix="lazyppl_checkout_", dir=checkout_parent)
        ).resolve()
        bin_dir = checkout_dir / "bin"
        bin_dir.mkdir()
        fake_stack = bin_dir / "stack"
        fake_stack.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f'if [ -e "{host_canary.name}" ]; then\n'
            '  echo "host /tmp leaked into sandbox" >&2\n'
            "  exit 91\n"
            "fi\n"
            "touch /tmp/sandbox_scratch_ok\n"
            'if [ "$1" = "repl" ]; then\n'
            "  load_line=$(cat)\n"
            '  src_file="${load_line#:load }"\n'
            '  test -r "$src_file"\n'
            '  grep -q "module " "$src_file"\n'
            '  if touch "${src_file}.ro_check" 2>/dev/null; then\n'
            '    echo "input dir was writable" >&2\n'
            "    exit 92\n"
            "fi\n"
            '  echo "Ok, 1 modules loaded."\n'
            "  exit 0\n"
            "fi\n"
            'if [ "$1" = "exec" ]; then\n'
            '  main_hs="${@: -1}"\n'
            '  src_dir=$(dirname "$main_hs")\n'
            '  test -r "$main_hs"\n'
            '  if [ -f "$src_dir/Submission.hs" ]; then\n'
            '    test -r "$src_dir/Submission.hs"\n'
            '    test -r "$src_dir/TaskData.hs"\n'
            '    echo \'{"log_lik":[[1.0,2.0],[1.5,2.5]],"n_draws":2,"n_obs":2}\'\n'
            "  else\n"
            '    echo "ok"\n'
            "  fi\n"
            "  exit 0\n"
            "fi\n"
            "exit 93\n",
            encoding="utf-8",
        )
        fake_stack.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}")

        try:
            executor = LazyPPLExecutor(lazyppl_dir=str(checkout_dir), compile_only=False)
            res = executor.execute(
                ToolCall(
                    name="lazyppl",
                    input={"code": 'module M where\nmain = putStrLn "ok"'},
                    raw="",
                ),
                timeout_s=5.0,
            )
            assert res.success is True, res.error
            assert res.output.strip() == "ok"

            res_loglik = executor.execute_for_loglik(
                "module Submission where\nmodel rows = return ()",
                [{"name": "x", "hs_type": "Double"}],
                [{"x": 1.0}, {"x": 2.0}],
                timeout_s=5.0,
            )
            assert res_loglik is not None
            assert res_loglik.shape == (2, 2)
        finally:
            os.unlink(host_canary.name)
            shutil.rmtree(checkout_dir, ignore_errors=True)

    @pytest.mark.parametrize(
        "unsafe_snippet",
        [
            'import System.IO\nmain = System.IO.writeFile "/tmp/marker" "x"',
            'main = writeFile "/tmp/marker" "x"',
            'import System.Process\nmain = callCommand "id"',
            "import System.IO.Unsafe (unsafePerformIO)\nx = unsafePerformIO (return 1)",
        ],
    )
    def test_lazyppl_blocks_unsafe_io_before_subprocess_dispatch(
        self, tmp_path, monkeypatch, unsafe_snippet
    ):
        from synthstats.core.types import ToolCall

        popen_calls: list[list[str]] = []
        monkeypatch.setattr(
            "subprocess.Popen",
            lambda cmd, *a, **kw: popen_calls.append(list(cmd)),
        )

        executor = LazyPPLExecutor(lazyppl_dir=str(tmp_path), compile_only=False)
        result = executor.execute(
            ToolCall(name="lazyppl", input={"code": unsafe_snippet}, raw=""),
            timeout_s=5.0,
        )
        assert result.success is False
        assert result.error is not None
        assert "Blocked" in result.error

        loglik = executor.execute_for_loglik(
            unsafe_snippet,
            [{"name": "x", "hs_type": "Double"}],
            [{"x": 1.0}],
        )
        assert loglik is None
        assert popen_calls == []


# ---------------------------------------------------------------------------
# End-to-end: real GHC execution (requires GHC + LazyPPL checkout)
# ---------------------------------------------------------------------------


def _has_ghc() -> bool:
    import shutil

    return shutil.which("ghc") is not None


_DUGONGS_SEED_PROGRAM = r"""module Submission
  ( State(..)
  , model
  , logLikRow
  ) where

import LazyPPL
import LazyPPL.Distributions
import TaskData (Row(..))
import LazyPPL.Distributions (normalLogPdf)

data State = State
  { alpha :: Double, betaParam :: Double
  , lambdaParam :: Double, sigma :: Double
  } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  a   <- sample $ normal 3 1
  b   <- sample $ normal 1 1
  lam <- sample $ beta 2 2
  s   <- sample $ gamma 2 0.5
  mapM_ (\r ->
    let predicted = a - b * (lam ** xVal r)
    in score $ normalPdf predicted s (yVal r)) rows
  return $ State a b lam s

logLikRow :: State -> Row -> Double
logLikRow (State a b lam s) r =
  let predicted = a - b * (lam ** xVal r)
  in normalLogPdf predicted s (yVal r)
"""


@pytest.mark.skipif(not _has_ghc(), reason="GHC not installed")
@pytest.mark.slow
class TestLazyPPLEndToEnd:
    """Requires GHC 9.4+ and a LazyPPL checkout configured with LAZYPPL_DEV_DIR."""

    def test_execute_for_loglik_real_ghc(self):
        """Full pipeline: Haskell code → runhaskell → JSON → numpy array."""
        from synthstats.tasks.statistical_modeling.envs.dugongs_env import DugongsEnv

        env = DugongsEnv()
        cols = env.get_lazyppl_columns()
        data = env.get_lazyppl_eval_data()

        executor = LazyPPLExecutor(compile_only=False)
        result = executor.execute_for_loglik(
            _DUGONGS_SEED_PROGRAM,
            cols,
            data,
            seed=42,
            burnin=100,
            thinning=5,
            ndraws=10,
            timeout_s=300,
        )

        assert result is not None, "execute_for_loglik returned None - check GHC/cabal setup"
        assert result.shape == (10, 10)  # ndraws=10, n_obs=10
        assert result.dtype == np.float64
        assert np.isfinite(result).all(), "log-lik matrix contains NaN/Inf"
