from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from synthstats.envs.task_runtime_env import TaskRuntimeEnv, TaskRuntimeEnvConfig
from synthstats.executors.elpd import ELPDRunner
from synthstats.judges.likelihood import LikelihoodJudge


def _elpd_payload(value: float) -> dict[str, object]:
    return {
        "elpd": value,
        "diagnostics": {"warning": False, "good_k": 0.7, "n_over_good_k": 0},
    }


def _make_boxing_env(
    *,
    judge: LikelihoodJudge | None,
    config: TaskRuntimeEnvConfig,
) -> TaskRuntimeEnv:
    return TaskRuntimeEnv(
        task=SimpleNamespace(name="dummy"),
        codec=SimpleNamespace(),
        executors={},
        judge=judge,
        config=config,
    )


def test_boxing_env_accepts_hydra_metadata_kwargs() -> None:
    env = TaskRuntimeEnv(
        task=SimpleNamespace(name="dummy"),
        codec=SimpleNamespace(),
        executors={},
        judge=None,
        config=TaskRuntimeEnvConfig(max_turns=7),
        name="dugongs",
        max_steps=20,
    )

    assert env.max_turns == 7


def test_score_program_uses_judge(monkeypatch) -> None:
    env = _make_boxing_env(
        judge=LikelihoodJudge(beta=0.5),
        config=TaskRuntimeEnvConfig(reward_floor=1e-4, reward_scale=1.0),
    )

    monkeypatch.setattr(ELPDRunner, "execute", lambda self, code: _elpd_payload(10.0))

    reward = TaskRuntimeEnv.score_program(env, "x = 1\n")
    assert reward == pytest.approx(math.exp(5.0))


def test_score_program_returns_floor_on_elpd_failure(monkeypatch) -> None:
    env = _make_boxing_env(
        judge=LikelihoodJudge(beta=0.5),
        config=TaskRuntimeEnvConfig(reward_floor=0.123, reward_scale=1.0),
    )

    monkeypatch.setattr(ELPDRunner, "execute", lambda self, code: {"error": "failed"})

    reward = TaskRuntimeEnv.score_program(env, "x = 1\n")
    assert reward == pytest.approx(0.123)


def test_score_program_reward_preserves_reward_metadata(monkeypatch) -> None:
    env = _make_boxing_env(
        judge=LikelihoodJudge(beta=0.5),
        config=TaskRuntimeEnvConfig(reward_floor=1e-4, reward_scale=2.0),
    )

    monkeypatch.setattr(
        ELPDRunner,
        "execute",
        lambda self, code: {
            "elpd": 2.0,
            "diagnostics": {"warning": False, "good_k": 0.7, "n_over_good_k": 0},
        },
    )

    reward = TaskRuntimeEnv.score_program_reward(env, "x = 1\n")

    assert reward.total == pytest.approx(2.0 * math.exp(1.0))
    assert reward.components == {"elpd": 2.0}
    assert reward.info["raw_total"] == pytest.approx(math.exp(1.0))
    assert reward.info["reward_scale"] == 2.0
    assert reward.info["diagnostics"] == {
        "warning": False,
        "good_k": 0.7,
        "n_over_good_k": 0,
    }
    assert reward.info["loo_status"] == "accepted"


def test_score_program_reward_uses_default_evaluation_seed(monkeypatch) -> None:
    env = _make_boxing_env(
        judge=LikelihoodJudge(beta=0.5),
        config=TaskRuntimeEnvConfig(reward_floor=1e-4, reward_scale=1.0),
    )
    captured: dict[str, object] = {}

    class _FakeELPDRunner:
        def __init__(
            self,
            *,
            timeout: float,
            random_seed: int | None = None,
            pool: object | None = None,
        ) -> None:
            captured["timeout"] = timeout
            captured["random_seed"] = random_seed
            captured["pool"] = pool

        def execute(self, program: str) -> dict[str, object]:
            assert program == "x = 1\n"
            return _elpd_payload(2.0)

    monkeypatch.setattr("synthstats.envs.task_runtime_env.ELPDRunner", _FakeELPDRunner)

    reward = TaskRuntimeEnv.score_program_reward(env, "x = 1\n")

    assert reward.total == pytest.approx(math.exp(1.0))
    assert captured["random_seed"] == 0


def test_score_program_without_judge_returns_floor(monkeypatch, caplog) -> None:
    env = _make_boxing_env(
        judge=None,
        config=TaskRuntimeEnvConfig(reward_floor=0.5, reward_scale=1.0),
    )

    monkeypatch.setattr(ELPDRunner, "execute", lambda self, code: _elpd_payload(10.0))

    with caplog.at_level("WARNING"):
        reward = TaskRuntimeEnv.score_program(env, "x = 1\n")

    assert reward == pytest.approx(0.5)
    assert any("score_program called without a Judge" in r.message for r in caplog.records)


def test_score_program_reward_without_judge_exposes_heuristic_floor_semantics(
    monkeypatch,
) -> None:
    env = _make_boxing_env(
        judge=None,
        config=TaskRuntimeEnvConfig(reward_floor=0.5, reward_scale=1.0),
    )

    monkeypatch.setattr(ELPDRunner, "execute", lambda self, code: _elpd_payload(10.0))

    reward = TaskRuntimeEnv.score_program_reward(env, "x = 1\n")

    assert reward.total == pytest.approx(0.5)
    assert reward.info["source"] == "no_judge"
    assert reward.info["reward_source"] == "no_judge"
    assert reward.info["evidence_kind"] == "heuristic"
    assert reward.info["truth_owner"] == "heuristic"


def test_score_program_reward_marks_unsupported_runtime_as_heuristic() -> None:
    env = _make_boxing_env(
        judge=LikelihoodJudge(beta=0.5),
        config=TaskRuntimeEnvConfig(reward_floor=0.25, reward_scale=1.0),
    )

    reward = TaskRuntimeEnv.score_program_reward(env, "x = 1\n", language="bogus")

    assert reward.total == pytest.approx(0.25)
    assert reward.info["source"] == "unsupported_runtime"
    assert reward.info["reward_source"] == "unsupported_runtime"
    assert reward.info["evidence_kind"] == "heuristic"
    assert reward.info["truth_owner"] == "heuristic"
    assert reward.info["language"] == "bogus"
