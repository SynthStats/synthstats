import math
import sys
import types

import pytest

from synthstats.core.types import Message, Reward, Trajectory


def make_trajectory(messages: list[Message] | None = None) -> Trajectory:
    if messages is None:
        messages = [Message(role="user", content="test")]
    return Trajectory(
        messages=messages,
        token_ids=[],
        token_logprobs=[],
        loss_mask=[],
        reward=Reward(total=0.0, components={}, info={}),
    )


class TestLikelihoodJudge:
    def test_likelihood_judge_with_zero_elpd(self):
        from synthstats.judges import LikelihoodJudge

        judge = LikelihoodJudge(beta=0.1)
        traj = make_trajectory()
        artifacts = {
            "diagnostics": {"warning": False, "good_k": 0.7, "n_over_good_k": 0},
            "elpd": 0.0,
        }

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        # exp(0.1 * 0) = exp(0) = 1.0
        assert reward.total == pytest.approx(1.0)
        assert "elpd" in reward.components
        assert reward.components["elpd"] == 0.0

    def test_likelihood_judge_with_positive_elpd(self):
        from synthstats.judges import LikelihoodJudge

        judge = LikelihoodJudge(beta=0.1)
        traj = make_trajectory()
        artifacts = {
            "diagnostics": {"warning": False, "good_k": 0.7, "n_over_good_k": 0},
            "elpd": 10.0,
        }

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        # exp(0.1 * 10) = exp(1.0) = e
        assert reward.total == pytest.approx(math.exp(1.0))

    def test_likelihood_judge_with_negative_elpd(self):
        from synthstats.judges import LikelihoodJudge

        judge = LikelihoodJudge(beta=0.1)
        traj = make_trajectory()
        artifacts = {
            "diagnostics": {"warning": False, "good_k": 0.7, "n_over_good_k": 0},
            "elpd": -10.0,
        }

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        # exp(0.1 * -10) = exp(-1.0)
        assert reward.total == pytest.approx(math.exp(-1.0))

    def test_likelihood_judge_clips_extreme_values(self):
        from synthstats.judges import LikelihoodJudge

        judge = LikelihoodJudge(beta=1.0, clip_range=(-10, 10))
        traj = make_trajectory()

        artifacts = {
            "diagnostics": {"warning": False, "good_k": 0.7, "n_over_good_k": 0},
            "elpd": 1000.0,
        }
        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)
        assert reward.total == pytest.approx(math.exp(10.0))

        artifacts = {
            "diagnostics": {"warning": False, "good_k": 0.7, "n_over_good_k": 0},
            "elpd": -1000.0,
        }
        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)
        assert reward.total == pytest.approx(math.exp(-10.0))

    def test_likelihood_judge_missing_elpd_returns_default(self):
        from synthstats.judges import LikelihoodJudge

        judge = LikelihoodJudge(beta=0.1)
        traj = make_trajectory()
        artifacts: dict[str, object] = {}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        # no evidence at all -> floor reward 0.0
        assert reward.total == 0.0
        assert reward.info["source"] == "validation.likelihood"
        assert reward.info["reward_source"] == "validation.likelihood"
        assert reward.info["evidence_kind"] == "verifier"
        assert reward.info["truth_owner"] == "verifier"
        assert reward.info["elpd_source"] == "default_no_evidence"

    def test_likelihood_judge_info_contains_log_reward(self):
        from synthstats.judges import LikelihoodJudge

        judge = LikelihoodJudge(beta=0.1)
        traj = make_trajectory()
        artifacts = {
            "diagnostics": {"warning": False, "good_k": 0.7, "n_over_good_k": 0},
            "elpd": 5.0,
        }

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert "log_reward" in reward.info
        assert reward.info["log_reward"] == pytest.approx(0.5)

    def test_likelihood_judge_nonfinite_mode_records_strict_shadow_rejection(self):
        from synthstats.judges import LikelihoodJudge

        judge = LikelihoodJudge(beta=0.1, loo_gate_mode="nonfinite")
        traj = make_trajectory()
        artifacts = {
            "elpd": 5.0,
            "diagnostics": {
                "warning": True,
                "good_k": 0.7,
                "n_over_good_k": 2,
            },
        }

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.total == pytest.approx(math.exp(0.5))
        assert reward.info["loo_status"] == "would_reject_strict"
        assert reward.info["would_reject_strict"] is True
        assert reward.info["diagnostics"]["warning"] is True

    def test_likelihood_judge_strict_mode_rejects_bad_diagnostics(self):
        from synthstats.judges import LikelihoodJudge

        judge = LikelihoodJudge(beta=0.1, loo_gate_mode="strict")
        traj = make_trajectory()
        artifacts = {
            "elpd": 5.0,
            "diagnostics": {
                "warning": False,
                "good_k": 0.7,
                "n_over_good_k": 1,
            },
        }

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.total == 0.0
        assert reward.components == {"elpd": 5.0}
        assert reward.info["loo_status"] == "rejected_strict"
        assert reward.info["would_reject_strict"] is True

    def test_likelihood_judge_default_direct_construction_is_strict(self):
        from synthstats.judges import LikelihoodJudge

        judge = LikelihoodJudge(beta=0.1)
        traj = make_trajectory()
        artifacts = {
            "elpd": 5.0,
            "diagnostics": {
                "warning": False,
                "good_k": 0.7,
                "n_over_good_k": 1,
            },
        }

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert judge.loo_gate_mode == "strict"
        assert reward.total == 0.0
        assert reward.info["loo_status"] == "rejected_strict"
        assert reward.info["would_reject_strict"] is True

    def test_likelihood_judge_prefers_runtime_primary_elpd_for_pymc(self, monkeypatch):
        from synthstats.judges import LikelihoodJudge

        judge = LikelihoodJudge(beta=0.1)
        traj = make_trajectory()

        def _unexpected_log_lik(_matrix):
            raise AssertionError("log_lik_matrix should not be used for pymc-primary scoring")

        monkeypatch.setattr(judge, "_elpd_from_log_lik_matrix", _unexpected_log_lik)

        reward = judge.score(
            task_name="test",
            trajectory=traj,
            artifacts={
                "diagnostics": {"warning": False, "good_k": 0.7, "n_over_good_k": 0},
                "language": "pymc",
                "elpd": 5.0,
                "log_lik_matrix": [[0.0]],
            },
        )

        assert reward.total == pytest.approx(math.exp(0.5))
        assert reward.info["elpd_source"] == "precomputed_runtime_primary"


class TestFormattingJudge:
    def test_formatting_judge_valid_program(self):
        from synthstats.judges import FormattingJudge

        judge = FormattingJudge()
        traj = make_trajectory()
        artifacts = {"program": "import pymc as pm\nx = pm.Normal('x', 0, 1)"}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.total == 1.0
        assert reward.info["is_valid"] is True
        assert reward.info["source"] == "validation.formatting"
        assert reward.info["reward_source"] == "validation.formatting"
        assert reward.info["evidence_kind"] == "verifier"
        assert reward.info["truth_owner"] == "verifier"

    def test_formatting_judge_detects_subprocess(self):
        from synthstats.judges import FormattingJudge

        judge = FormattingJudge()
        traj = make_trajectory()
        artifacts = {"program": "import subprocess\nsubprocess.run(['ls'])"}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.total == 0.0
        assert reward.info["is_valid"] is False

    def test_formatting_judge_detects_socket(self):
        from synthstats.judges import FormattingJudge

        judge = FormattingJudge()
        traj = make_trajectory()
        artifacts = {"program": "import socket\nsocket.socket()"}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.total == 0.0

    def test_formatting_judge_detects_os_system(self):
        from synthstats.judges import FormattingJudge

        judge = FormattingJudge()
        traj = make_trajectory()
        artifacts = {"program": "import os\nos.system('rm -rf /')"}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.total == 0.0

    def test_formatting_judge_custom_forbidden_imports(self):
        from synthstats.judges import FormattingJudge

        judge = FormattingJudge(forbidden_imports=["pandas", "numpy"])
        traj = make_trajectory()

        artifacts = {"program": "import pandas as pd"}
        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)
        assert reward.total == 0.0

        artifacts = {"program": "import pymc as pm"}
        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)
        assert reward.total == 1.0

    def test_formatting_judge_empty_program(self):
        from synthstats.judges import FormattingJudge

        judge = FormattingJudge()
        traj = make_trajectory()
        artifacts = {"program": ""}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.total == 1.0

    def test_formatting_judge_missing_program(self):
        from synthstats.judges import FormattingJudge

        judge = FormattingJudge()
        traj = make_trajectory()
        artifacts: dict[str, object] = {}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.total == 1.0


class TestCompositeJudge:
    def test_composite_judge_empty_list(self):
        from synthstats.judges import CompositeJudge

        judge = CompositeJudge([])
        traj = make_trajectory()

        reward = judge.score(task_name="test", trajectory=traj, artifacts={})

        assert reward.total == 0.0
        assert reward.components == {}
        assert reward.info["source"] == "composite"
        assert reward.info["reward_source"] == "composite"
        assert reward.info["evidence_kind"] == "heuristic"
        assert reward.info["truth_owner"] == "heuristic"

    def test_composite_judge_single_judge(self):
        from synthstats.judges import CompositeJudge, LikelihoodJudge

        likelihood = LikelihoodJudge(beta=0.1)
        judge = CompositeJudge([(likelihood, 1.0)])
        traj = make_trajectory()
        artifacts = {
            "diagnostics": {"warning": False, "good_k": 0.7, "n_over_good_k": 0},
            "elpd": 0.0,
        }

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        # weight 1.0 * exp(0) = 1.0
        assert reward.total == pytest.approx(1.0)
        assert "LikelihoodJudge" in reward.components

    def test_composite_judge_weighted_combination(self):
        from synthstats.judges import CompositeJudge, FormattingJudge, LikelihoodJudge

        likelihood = LikelihoodJudge(beta=0.1)
        formatting = FormattingJudge()

        judge = CompositeJudge(
            [
                (likelihood, 0.7),
                (formatting, 0.3),
            ],
            scalarization_mode="weighted_sum",
        )
        traj = make_trajectory()
        artifacts = {
            "diagnostics": {"warning": False, "good_k": 0.7, "n_over_good_k": 0},
            "elpd": 0.0,
            "program": "x = 1",
        }

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        # 0.7 * 1.0 + 0.3 * 1.0 = 1.0
        expected = 0.7 * 1.0 + 0.3 * 1.0
        assert reward.total == pytest.approx(expected)
        assert "LikelihoodJudge" in reward.components
        assert "FormattingJudge" in reward.components

    def test_composite_judge_with_failing_format(self):
        from synthstats.judges import CompositeJudge, FormattingJudge, LikelihoodJudge

        likelihood = LikelihoodJudge(beta=0.1)
        formatting = FormattingJudge()

        judge = CompositeJudge(
            [
                (likelihood, 0.7),
                (formatting, 0.3),
            ],
            scalarization_mode="weighted_sum",
        )
        traj = make_trajectory()
        artifacts = {
            "diagnostics": {"warning": False, "good_k": 0.7, "n_over_good_k": 0},
            "elpd": 0.0,
            "program": "import subprocess",
        }

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        # 0.7 * 1.0 + 0.3 * 0.0 = 0.7
        expected = 0.7 * 1.0 + 0.3 * 0.0
        assert reward.total == pytest.approx(expected)

    def test_composite_judge_default_uses_weighted_log_product(self):
        from synthstats.core.judge import Judge
        from synthstats.judges import CompositeJudge

        class _LowJudge:
            def score(self, *, task_name: str, trajectory: Trajectory, artifacts: dict) -> Reward:
                del task_name, trajectory, artifacts
                return Reward(total=0.25, components={}, info={})

        class _HighJudge:
            def score(self, *, task_name: str, trajectory: Trajectory, artifacts: dict) -> Reward:
                del task_name, trajectory, artifacts
                return Reward(total=4.0, components={}, info={})

        assert isinstance(_LowJudge(), Judge)
        assert isinstance(_HighJudge(), Judge)

        judge = CompositeJudge(
            [
                (_LowJudge(), 0.5),
                (_HighJudge(), 0.5),
            ]
        )
        traj = make_trajectory()

        reward = judge.score(task_name="test", trajectory=traj, artifacts={})

        assert reward.total == pytest.approx(1.0)
        assert reward.info["scalarization_mode"] == "weighted_log_product"
        assert reward.info["factor_weights"] == {
            "_LowJudge": 0.5,
            "_HighJudge": 0.5,
        }

    def test_composite_judge_components_track_individual_scores(self):
        from synthstats.judges import CompositeJudge, FormattingJudge, LikelihoodJudge

        likelihood = LikelihoodJudge(beta=0.1)
        formatting = FormattingJudge()

        judge = CompositeJudge(
            [
                (likelihood, 0.5),
                (formatting, 0.5),
            ]
        )
        traj = make_trajectory()
        artifacts = {
            "diagnostics": {"warning": False, "good_k": 0.7, "n_over_good_k": 0},
            "elpd": 10.0,
            "program": "x = 1",
        }

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.components["LikelihoodJudge"] == pytest.approx(math.exp(1.0))
        assert reward.components["FormattingJudge"] == 1.0

    def test_composite_judge_implements_protocol(self):
        from synthstats.core.judge import Judge
        from synthstats.judges import CompositeJudge

        judge = CompositeJudge([])
        assert isinstance(judge, Judge)

    def test_composite_judge_weighted_log_product_clamps_nonpositive(self):
        from synthstats.judges import CompositeJudge

        class _ZeroJudge:
            def score(self, *, task_name, trajectory, artifacts):
                return Reward(total=0.0, components={}, info={})

        class _NegJudge:
            def score(self, *, task_name, trajectory, artifacts):
                return Reward(total=-1.0, components={}, info={})

        class _PosJudge:
            def score(self, *, task_name, trajectory, artifacts):
                return Reward(total=2.0, components={}, info={})

        traj = make_trajectory()
        min_factor = 1e-12

        judge_zero = CompositeJudge(
            [(_ZeroJudge(), 0.5), (_PosJudge(), 0.5)],
            min_factor=min_factor,
        )
        reward_zero = judge_zero.score(task_name="t", trajectory=traj, artifacts={})
        expected_zero = math.exp(0.5 * math.log(min_factor) + 0.5 * math.log(2.0))
        assert math.isfinite(reward_zero.total)
        assert reward_zero.total == pytest.approx(expected_zero)

        judge_neg = CompositeJudge(
            [(_NegJudge(), 0.5), (_PosJudge(), 0.5)],
            min_factor=min_factor,
        )
        reward_neg = judge_neg.score(task_name="t", trajectory=traj, artifacts={})
        expected_neg = math.exp(0.5 * math.log(min_factor) + 0.5 * math.log(2.0))
        assert math.isfinite(reward_neg.total)
        assert reward_neg.total == pytest.approx(expected_neg)

    def test_composite_judge_survives_sub_judge_exception(self):
        from synthstats.judges import CompositeJudge

        class _CrashingJudge:
            def score(self, *, task_name, trajectory, artifacts):
                raise RuntimeError("API key missing")

        class _GoodJudge:
            def score(self, *, task_name, trajectory, artifacts):
                return Reward(total=0.8, components={}, info={})

        traj = make_trajectory()

        judge = CompositeJudge(
            [(_GoodJudge(), 0.6), (_CrashingJudge(), 0.4)],
            scalarization_mode="weighted_sum",
        )
        reward = judge.score(task_name="t", trajectory=traj, artifacts={})
        assert reward.total == pytest.approx(0.6 * 0.8)
        assert reward.components["_GoodJudge"] == 0.8
        assert reward.components["_CrashingJudge"] == 0.0
        assert reward.factors["_CrashingJudge"] == 0.0
        assert "errors" in reward.info
        assert "_CrashingJudge" in reward.info["errors"]
        assert "RuntimeError" in reward.info["errors"]["_CrashingJudge"]

    def test_composite_judge_all_judges_crash_retains_zero_factors(self):
        from synthstats.judges import CompositeJudge

        class _CrashingJudge:
            def score(self, *, task_name, trajectory, artifacts):
                raise ValueError("boom")

        traj = make_trajectory()
        judge = CompositeJudge(
            [(_CrashingJudge(), 0.5), (_CrashingJudge(), 0.5)],
        )
        reward = judge.score(task_name="t", trajectory=traj, artifacts={})
        assert reward.total == pytest.approx(1e-12)
        assert list(reward.factors.values()) == [0.0, 0.0]


class TestLLMCriticJudge:
    def test_llm_critic_judge_initialization(self):
        from synthstats.judges import LLMCriticJudge

        judge = LLMCriticJudge()
        assert judge.model_name == "gpt-4o-mini"
        assert judge.temperature == 0.3
        assert judge.num_samples == 1
        assert judge.default_score == 0.5

    def test_llm_critic_judge_custom_initialization(self):
        from synthstats.judges import LLMCriticJudge
        from synthstats.judges.llm_critic import LLMCriticConfig

        judge = LLMCriticJudge(
            LLMCriticConfig(
                model_name="claude-3-haiku-20240307",
                temperature=0.5,
                num_samples=3,
                default_score=0.3,
                api_base="https://api.example.com/v1",
                api_key_env_var="FAKE_LLM_KEY",
                max_tokens=512,
                timeout_s=12.0,
                max_retries=4,
            )
        )
        assert judge.model_name == "claude-3-haiku-20240307"
        assert judge.temperature == 0.5
        assert judge.num_samples == 3
        assert judge.default_score == 0.3
        assert judge.api_base == "https://api.example.com/v1"
        assert judge.api_key_env_var == "FAKE_LLM_KEY"
        assert judge.max_tokens == 512
        assert judge.timeout_s == 12.0
        assert judge.max_retries == 4

    def test_llm_critic_judge_builds_prompt(self):
        from synthstats.judges import LLMCriticJudge

        judge = LLMCriticJudge()
        program = "import pymc as pm\nx = pm.Normal('x', 0, 1)"
        task_name = "dugongs"

        prompt = judge._build_prompt(program, task_name)

        assert "dugongs" in prompt
        assert "import pymc as pm" in prompt
        assert "code_quality" in prompt
        assert "statistical_validity" in prompt
        assert "problem_alignment" in prompt

    def test_llm_critic_judge_builds_prompt_custom_template(self):
        from synthstats.judges import LLMCriticJudge
        from synthstats.judges.llm_critic import LLMCriticConfig

        custom_template = "Task: {task_name}\nCode:\n{program}\nRate it."
        judge = LLMCriticJudge(LLMCriticConfig(prompt_template=custom_template))

        prompt = judge._build_prompt("x = 1", "test_task")

        assert prompt == "Task: test_task\nCode:\nx = 1\nRate it."

    def test_llm_critic_judge_parses_critique_valid(self):
        from synthstats.judges import LLMCriticJudge

        judge = LLMCriticJudge()
        response = """SCORES:
code_quality: 0.8
statistical_validity: 0.7
problem_alignment: 0.9
RATIONALE: Good model structure with appropriate priors."""

        scores = judge._parse_critique(response)

        assert scores["code_quality"] == pytest.approx(0.8)
        assert scores["statistical_validity"] == pytest.approx(0.7)
        assert scores["problem_alignment"] == pytest.approx(0.9)

    def test_llm_critic_judge_parses_critique_clamps_values(self):
        from synthstats.judges import LLMCriticJudge

        judge = LLMCriticJudge()
        response = """SCORES:
code_quality: 1.5
statistical_validity: -0.2
problem_alignment: 0.5
RATIONALE: Testing edge cases."""

        scores = judge._parse_critique(response)

        assert scores["code_quality"] == 1.0  # clamped from 1.5
        assert scores["statistical_validity"] == 0.0  # clamped from -0.2
        assert scores["problem_alignment"] == 0.5

    def test_llm_critic_judge_parses_critique_missing_scores(self):
        from synthstats.judges import LLMCriticJudge
        from synthstats.judges.llm_critic import LLMCriticConfig

        judge = LLMCriticJudge(LLMCriticConfig(default_score=0.5))
        response = """SCORES:
code_quality: 0.8
RATIONALE: Partial response."""

        scores = judge._parse_critique(response)

        assert scores["code_quality"] == pytest.approx(0.8)
        assert scores["statistical_validity"] == 0.5  # default
        assert scores["problem_alignment"] == 0.5  # default

    def test_llm_critic_judge_parses_critique_malformed(self):
        from synthstats.judges import LLMCriticJudge
        from synthstats.judges.llm_critic import LLMCriticConfig

        judge = LLMCriticJudge(LLMCriticConfig(default_score=0.5))
        response = "This is not in the expected format at all."

        scores = judge._parse_critique(response)

        assert scores["code_quality"] == 0.5
        assert scores["statistical_validity"] == 0.5
        assert scores["problem_alignment"] == 0.5

    def test_llm_critic_judge_parses_critique_case_insensitive(self):
        from synthstats.judges import LLMCriticJudge

        judge = LLMCriticJudge()
        response = """scores:
CODE_QUALITY: 0.6
Statistical_Validity: 0.7
PROBLEM_ALIGNMENT: 0.8
rationale: Case variations."""

        scores = judge._parse_critique(response)

        assert scores["code_quality"] == pytest.approx(0.6)
        assert scores["statistical_validity"] == pytest.approx(0.7)
        assert scores["problem_alignment"] == pytest.approx(0.8)

    def test_llm_critic_judge_score_empty_program(self):
        from synthstats.judges import LLMCriticJudge

        judge = LLMCriticJudge()
        traj = make_trajectory()
        artifacts = {"program": ""}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.total == 0.0
        assert reward.components["code_quality"] == 0.0
        assert reward.info["source"] == "llm_critic"
        assert reward.info["reward_source"] == "llm_critic"
        assert reward.info["evidence_kind"] == "heuristic"
        assert reward.info["truth_owner"] == "heuristic"
        assert reward.info.get("error") == "empty_program"

    def test_llm_critic_judge_score_missing_program(self):
        from synthstats.judges import LLMCriticJudge

        judge = LLMCriticJudge()
        traj = make_trajectory()
        artifacts: dict[str, object] = {}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.total == 0.0
        assert reward.info.get("error") == "empty_program"

    def test_llm_critic_judge_score_without_api(self):
        from unittest.mock import patch

        from synthstats.judges import LLMCriticJudge

        judge = LLMCriticJudge()
        traj = make_trajectory()
        artifacts = {"program": "import pymc as pm\nx = pm.Normal('x', 0, 1)"}

        mock_response = """SCORES:
code_quality: 0.8
statistical_validity: 0.9
problem_alignment: 0.7
RATIONALE: Good basic model."""

        with patch.object(judge, "_call_llm", return_value=mock_response):
            reward = judge.score(task_name="dugongs", trajectory=traj, artifacts=artifacts)

        assert reward.total == pytest.approx((0.8 + 0.9 + 0.7) / 3)
        assert reward.components["code_quality"] == pytest.approx(0.8)
        assert reward.components["statistical_validity"] == pytest.approx(0.9)
        assert reward.components["problem_alignment"] == pytest.approx(0.7)
        assert reward.info["num_samples"] == 1

    def test_llm_critic_judge_score_api_failure(self):
        from unittest.mock import patch

        from synthstats.judges import LLMCriticJudge
        from synthstats.judges.llm_critic import LLMCriticConfig, LLMCriticUnavailableError

        judge = LLMCriticJudge(LLMCriticConfig(default_score=0.5))
        traj = make_trajectory()
        artifacts = {"program": "x = 1"}

        with patch.object(judge, "_call_llm", return_value=None):
            with pytest.raises(LLMCriticUnavailableError, match="LLM critic returned no critique"):
                judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

    def test_llm_critic_judge_score_multiple_samples(self):
        from unittest.mock import patch

        from synthstats.judges import LLMCriticJudge
        from synthstats.judges.llm_critic import LLMCriticConfig

        judge = LLMCriticJudge(LLMCriticConfig(num_samples=2))
        traj = make_trajectory()
        artifacts = {"program": "x = 1"}

        responses = [
            "SCORES:\ncode_quality: 0.6\nstatistical_validity: 0.8\nproblem_alignment: 0.7",
            "SCORES:\ncode_quality: 0.8\nstatistical_validity: 0.6\nproblem_alignment: 0.9",
        ]
        call_count = [0]

        def mock_call_llm(prompt):
            idx = call_count[0]
            call_count[0] += 1
            return responses[idx] if idx < len(responses) else None

        with patch.object(judge, "_call_llm", side_effect=mock_call_llm):
            reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.components["code_quality"] == pytest.approx((0.6 + 0.8) / 2)
        assert reward.components["statistical_validity"] == pytest.approx((0.8 + 0.6) / 2)
        assert reward.components["problem_alignment"] == pytest.approx((0.7 + 0.9) / 2)
        assert reward.info["num_samples"] == 2

    def test_llm_critic_judge_call_llm_passes_provider_settings(self, monkeypatch):
        from synthstats.judges import LLMCriticJudge

        captured: dict[str, object] = {}

        def _completion(**kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]
            )

        monkeypatch.setenv("FAKE_LLM_KEY", "fake-key-1")
        monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))

        from synthstats.judges.llm_critic import LLMCriticConfig

        judge = LLMCriticJudge(
            LLMCriticConfig(
                model_name="openai/gpt-5-mini",
                api_base="https://proxy.example.com/v1",
                api_key_env_var="FAKE_LLM_KEY",
                max_tokens=321,
                timeout_s=9.0,
                max_retries=2,
                temperature=0.1,
            )
        )

        response = judge._call_llm("hello")

        assert response == "ok"
        assert captured["model"] == "openai/gpt-5-mini"
        assert captured["api_base"] == "https://proxy.example.com/v1"
        assert captured["api_key"] == "fake-key-1"
        assert captured["max_tokens"] == 321
        assert captured["timeout"] == 9.0
        assert captured["num_retries"] == 2

    def test_llm_critic_judge_num_samples_minimum(self):
        from synthstats.judges import LLMCriticJudge
        from synthstats.judges.llm_critic import LLMCriticConfig

        judge = LLMCriticJudge(LLMCriticConfig(num_samples=0))
        assert judge.num_samples == 1

        judge = LLMCriticJudge(LLMCriticConfig(num_samples=-5))
        assert judge.num_samples == 1

    def test_llm_critic_judge_in_composite(self):
        from unittest.mock import patch

        from synthstats.judges import CompositeJudge, FormattingJudge, LLMCriticJudge

        llm_critic = LLMCriticJudge()
        formatting = FormattingJudge()

        judge = CompositeJudge(
            [
                (llm_critic, 0.6),
                (formatting, 0.4),
            ],
            scalarization_mode="weighted_sum",
        )

        traj = make_trajectory()
        artifacts = {"program": "import pymc as pm\nx = pm.Normal('x', 0, 1)"}

        mock_response = """SCORES:
code_quality: 0.9
statistical_validity: 0.9
problem_alignment: 0.9
RATIONALE: Great model."""

        with patch.object(llm_critic, "_call_llm", return_value=mock_response):
            reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        # 0.6 * 0.9 (llm avg) + 0.4 * 1.0 (formatting) = 0.54 + 0.4 = 0.94
        expected = 0.6 * 0.9 + 0.4 * 1.0
        assert reward.total == pytest.approx(expected)
        assert "LLMCriticJudge" in reward.components
        assert "FormattingJudge" in reward.components


class TestCompilationJudge:
    def test_compilation_judge_empty_program(self):
        from synthstats.judges import CompilationJudge

        judge = CompilationJudge()
        traj = make_trajectory()
        artifacts = {"program": "", "language": "lazyppl"}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.total == 0.0
        assert reward.components["compiles"] == 0.0
        assert reward.info["source"] == "validation.compilation"
        assert reward.info["reward_source"] == "validation.compilation"
        assert reward.info["evidence_kind"] == "verifier"
        assert reward.info["truth_owner"] == "verifier"
        assert reward.info["error"] == "empty_program"

    def test_compilation_judge_missing_program(self):
        from synthstats.judges import CompilationJudge

        judge = CompilationJudge()
        traj = make_trajectory()
        artifacts = {"language": "lazyppl"}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.total == 0.0

    def test_compilation_judge_valid_python(self):
        from synthstats.judges import CompilationJudge

        judge = CompilationJudge()
        traj = make_trajectory()
        artifacts = {"program": "import pymc as pm\nx = pm.Normal('x', 0, 1)", "language": "pymc"}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.total == 1.0
        assert reward.components["compiles"] == 1.0

    def test_compilation_judge_invalid_python(self):
        from synthstats.judges import CompilationJudge

        judge = CompilationJudge()
        traj = make_trajectory()
        artifacts = {"program": "def f(\n  broken syntax", "language": "pymc"}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.total == 0.0
        assert reward.components["compiles"] == 0.0

    def test_compilation_judge_defaults_to_pymc(self):
        from synthstats.judges import CompilationJudge

        judge = CompilationJudge()
        traj = make_trajectory()
        artifacts = {"program": "x = 1"}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.total == 1.0
        assert reward.info["language"] == "pymc"

    def test_compilation_judge_implements_protocol(self):
        from synthstats.core.judge import Judge
        from synthstats.judges import CompilationJudge

        judge = CompilationJudge()
        assert isinstance(judge, Judge)

    def test_compilation_judge_in_composite(self):
        from synthstats.judges import CompilationJudge, CompositeJudge, FormattingJudge

        compilation = CompilationJudge()
        formatting = FormattingJudge()

        judge = CompositeJudge(
            [
                (compilation, 0.7),
                (formatting, 0.3),
            ],
            scalarization_mode="weighted_sum",
        )
        traj = make_trajectory()
        artifacts = {"program": "x = 1", "language": "pymc"}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        # 0.7 * 1.0 (compiles) + 0.3 * 1.0 (valid format) = 1.0
        assert reward.total == pytest.approx(1.0)
        assert "CompilationJudge" in reward.components
        assert "FormattingJudge" in reward.components


class TestLLMCriticJudgeLanguageTemplate:
    def test_lazyppl_language_uses_lazyppl_template(self):
        from synthstats.judges import LLMCriticJudge
        from synthstats.judges.llm_critic import LLMCriticConfig

        judge = LLMCriticJudge(LLMCriticConfig(language="lazyppl"))

        assert "LazyPPL" in judge.prompt_template or "Haskell" in judge.prompt_template
        assert "PyMC" not in judge.prompt_template

    def test_pymc_language_uses_default_template(self):
        from synthstats.judges import LLMCriticJudge
        from synthstats.judges.llm_critic import LLMCriticConfig

        judge = LLMCriticJudge(LLMCriticConfig(language="pymc"))

        assert "PyMC" in judge.prompt_template


class TestCompilationPlusLLMCriticComposite:
    def test_weighted_sum_matches_lazyppl_prod_config(self):
        from unittest.mock import patch

        from synthstats.judges import (
            CompilationJudge,
            CompositeJudge,
            LikelihoodJudge,
            LLMCriticJudge,
        )
        from synthstats.judges.llm_critic import LLMCriticConfig

        compilation = CompilationJudge()
        likelihood = LikelihoodJudge(beta=0.1, loo_gate_mode="strict")
        llm_critic = LLMCriticJudge(LLMCriticConfig(language="lazyppl"))

        judge = CompositeJudge(
            [
                (compilation, 0.4),
                (likelihood, 0.4),
                (llm_critic, 0.2),
            ],
            scalarization_mode="weighted_sum",
        )

        traj = make_trajectory()
        artifacts = {
            "program": "model = pure 1",
            "language": "lazyppl",
            "elpd": 5.0,
            "diagnostics": {
                "warning": False,
                "good_k": 0.7,
                "n_over_good_k": 0,
            },
        }

        mock_response = """SCORES:
code_quality: 0.8
statistical_validity: 0.6
problem_alignment: 0.7
RATIONALE: Decent model."""

        mocked_score = (0.8 + 0.6 + 0.7) / 3

        with (
            patch.object(compilation, "_check_lazyppl", return_value=(True, None)),
            patch.object(llm_critic, "_call_llm", return_value=mock_response),
        ):
            reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        expected = 0.4 * 1.0 + 0.4 * math.exp(0.5) + 0.2 * mocked_score
        assert reward.total == pytest.approx(expected)
        assert "CompilationJudge" in reward.components
        assert reward.components["LikelihoodJudge"] == pytest.approx(math.exp(0.5))
        assert "LLMCriticJudge" in reward.components


class TestJudgeExports:
    def test_judges_in_all(self):
        import synthstats.judges

        assert "CompilationJudge" in synthstats.judges.__all__
        assert "CompositeJudge" in synthstats.judges.__all__
        assert "LikelihoodJudge" in synthstats.judges.__all__
        assert "FormattingJudge" in synthstats.judges.__all__
        assert "LLMCriticJudge" in synthstats.judges.__all__
