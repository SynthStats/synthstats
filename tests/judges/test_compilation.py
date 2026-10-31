"""Tests for synthstats.judges.compilation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from synthstats.core.types import Message, Reward, Trajectory
from synthstats.judges.compilation import CompilationJudge


def _dummy_trajectory() -> Trajectory:
    return Trajectory(
        messages=[Message(role="user", content="test")],
        token_ids=[[1]],
        token_logprobs=[[0.0]],
        loss_mask=[[True]],
        reward=Reward(total=0.0, components={}, info={}),
    )


@pytest.fixture
def judge() -> CompilationJudge:
    return CompilationJudge()


@pytest.fixture
def traj() -> Trajectory:
    return _dummy_trajectory()


class TestPythonCompilation:
    def test_valid_python_scores_one(self, judge: CompilationJudge, traj: Trajectory):
        artifacts = {"program": "x = 1 + 2\nprint(x)", "language": "pymc"}
        result = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)
        assert result.total == 1.0
        assert result.components["compiles"] == 1.0

    def test_syntax_error_scores_zero(self, judge: CompilationJudge, traj: Trajectory):
        artifacts = {"program": "def f(\n", "language": "pymc"}
        result = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)
        assert result.total == 0.0
        assert result.components["compiles"] == 0.0
        assert "error" in result.info

    def test_empty_string_scores_zero(self, judge: CompilationJudge, traj: Trajectory):
        artifacts = {"program": "", "language": "pymc"}
        result = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)
        assert result.total == 0.0
        assert result.info["error"] == "empty_program"

    def test_whitespace_only_scores_zero(self, judge: CompilationJudge, traj: Trajectory):
        artifacts = {"program": "   \n\t  ", "language": "pymc"}
        result = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)
        assert result.total == 0.0
        assert result.info["error"] == "empty_program"

    def test_missing_program_key_scores_zero(self, judge: CompilationJudge, traj: Trajectory):
        result = judge.score(task_name="test", trajectory=traj, artifacts={})
        assert result.total == 0.0
        assert result.info["error"] == "empty_program"

    def test_multiline_valid_python(self, judge: CompilationJudge, traj: Trajectory):
        code = "import math\n\ndef area(r):\n    return math.pi * r ** 2\n"
        artifacts = {"program": code, "language": "pymc"}
        result = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)
        assert result.total == 1.0

    def test_indentation_error_scores_zero(self, judge: CompilationJudge, traj: Trajectory):
        code = "def f():\nx = 1"
        artifacts = {"program": code, "language": "pymc"}
        result = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)
        assert result.total == 0.0

    @pytest.mark.parametrize("code", ["return 1\n", "break\n", "nonlocal x\n"])
    def test_uncompilable_ast_valid_python_scores_zero(
        self, judge: CompilationJudge, traj: Trajectory, code: str
    ):
        artifacts = {"program": code, "language": "pymc"}
        result = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)
        assert result.total == 0.0
        assert result.components["compiles"] == 0.0
        assert "error" in result.info


class TestLanguageDefault:
    def test_defaults_to_pymc(self, judge: CompilationJudge, traj: Trajectory):
        artifacts = {"program": "x = 1"}
        result = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)
        assert result.total == 1.0
        assert result.info["language"] == "pymc"

    def test_haskell_alias_routes_to_lazyppl(self, judge: CompilationJudge, traj: Trajectory):
        judge._check_lazyppl = MagicMock(return_value=(True, None))
        judge._check_python = MagicMock(return_value=(False, "should_not_run"))

        result = judge.score(
            task_name="test",
            trajectory=traj,
            artifacts={"program": "model = 1", "language": "haskell"},
        )

        assert result.total == 1.0
        assert result.info["language"] == "lazyppl"
        judge._check_lazyppl.assert_called_once_with("model = 1")
        judge._check_python.assert_not_called()

    def test_unknown_language_falls_back_to_python_check(
        self, judge: CompilationJudge, traj: Trajectory
    ):
        judge._check_lazyppl = MagicMock(return_value=(False, "should_not_run"))
        judge._check_python = MagicMock(return_value=(True, None))

        result = judge.score(
            task_name="test",
            trajectory=traj,
            artifacts={"program": "x = 1", "language": "custom_python"},
        )

        assert result.total == 1.0
        assert result.info["language"] == "custom_python"
        judge._check_lazyppl.assert_not_called()
        judge._check_python.assert_called_once_with("x = 1")


class TestRewardStructure:
    def test_success_info_has_language(self, judge: CompilationJudge, traj: Trajectory):
        artifacts = {"program": "x = 1", "language": "pymc"}
        result = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)
        assert result.info["language"] == "pymc"
        assert "error" not in result.info

    def test_failure_info_has_error_and_language(self, judge: CompilationJudge, traj: Trajectory):
        artifacts = {"program": "x ==== y", "language": "pymc"}
        result = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)
        assert result.info["language"] == "pymc"
        assert "error" in result.info


class TestCompilationJudgeInit:
    def test_default_timeout(self):
        j = CompilationJudge()
        assert j.timeout_s == 60.0

    def test_custom_timeout(self):
        j = CompilationJudge(timeout_s=10.0)
        assert j.timeout_s == 10.0

    def test_lazyppl_executor_lazy_init(self):
        j = CompilationJudge()
        assert j._lazyppl_executor is None
