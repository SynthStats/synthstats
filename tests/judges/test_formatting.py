from __future__ import annotations

from synthstats.core.types import Message, Reward, Trajectory
from synthstats.judges.formatting import FormattingJudge


def _traj() -> Trajectory:
    return Trajectory(
        messages=[Message(role="user", content="test")],
        token_ids=[],
        token_logprobs=[],
        loss_mask=[],
        reward=Reward(total=0.0, components={}, info={}),
    )


class TestFormattingJudgeDefaults:
    def test_default_forbidden_imports(self):
        judge = FormattingJudge()
        assert "subprocess" in judge.forbidden_imports
        assert "socket" in judge.forbidden_imports
        assert "os.system" in judge.forbidden_imports

    def test_none_forbidden_imports_uses_defaults(self):
        judge = FormattingJudge(forbidden_imports=None)
        assert len(judge.forbidden_imports) == 3


class TestFormattingJudgeScoring:
    def test_clean_pymc_program_passes(self):
        judge = FormattingJudge()
        code = "import pymc as pm\nwith pm.Model():\n    x = pm.Normal('x', 0, 1)"
        artifacts = {"program": code}
        reward = judge.score(task_name="t", trajectory=_traj(), artifacts=artifacts)
        assert reward.total == 1.0

    def test_multiple_forbidden_imports_caught(self):
        judge = FormattingJudge()
        code = "import subprocess\nimport socket"
        artifacts = {"program": code}
        reward = judge.score(task_name="t", trajectory=_traj(), artifacts=artifacts)
        assert reward.total == 0.0

    def test_forbidden_string_in_comment_still_fails(self):
        judge = FormattingJudge()
        code = "# uses subprocess internally\nx = 1"
        artifacts = {"program": code}
        reward = judge.score(task_name="t", trajectory=_traj(), artifacts=artifacts)
        assert reward.total == 0.0

    def test_partial_match_triggers_failure(self):
        judge = FormattingJudge()
        code = "os.system('echo hi')"
        artifacts = {"program": code}
        reward = judge.score(task_name="t", trajectory=_traj(), artifacts=artifacts)
        assert reward.total == 0.0

    def test_empty_forbidden_list_falls_back_to_defaults(self):
        judge = FormattingJudge(forbidden_imports=[])
        assert len(judge.forbidden_imports) == 3
        code = "import subprocess"
        artifacts = {"program": code}
        reward = judge.score(task_name="t", trajectory=_traj(), artifacts=artifacts)
        assert reward.total == 0.0

    def test_reward_components_are_empty(self):
        judge = FormattingJudge()
        artifacts = {"program": "x = 1"}
        reward = judge.score(task_name="t", trajectory=_traj(), artifacts=artifacts)
        assert reward.components == {}

    def test_missing_program_key_treated_as_empty_string(self):
        judge = FormattingJudge()
        reward = judge.score(task_name="t", trajectory=_traj(), artifacts={})
        assert reward.total == 1.0
        assert reward.info["is_valid"] is True

    def test_custom_single_forbidden_import(self):
        judge = FormattingJudge(forbidden_imports=["torch"])
        artifacts = {"program": "import torch\nx = torch.randn(10)"}
        reward = judge.score(task_name="t", trajectory=_traj(), artifacts=artifacts)
        assert reward.total == 0.0

        artifacts_ok = {"program": "import pymc as pm"}
        reward_ok = judge.score(task_name="t", trajectory=_traj(), artifacts=artifacts_ok)
        assert reward_ok.total == 1.0
