from __future__ import annotations

import logging

import pytest

from synthstats.core.judge_runtime import (
    build_stub_trajectory,
    record_reward_details,
    score_stubbed_messages,
    score_stubbed_messages_reward,
    score_stubbed_messages_total,
)
from synthstats.core.types import Message, Reward


class _RecordingJudge:
    def __init__(self, reward: Reward) -> None:
        self.reward = reward
        self.calls: list[tuple[str, list[Message], dict[str, object]]] = []

    def score(self, *, task_name, trajectory, artifacts):
        self.calls.append((task_name, trajectory.messages, dict(artifacts)))
        return self.reward


def test_build_stub_trajectory_accepts_message_dicts_and_messages() -> None:
    trajectory = build_stub_trajectory(
        [
            {"role": "user", "content": "prompt"},
            Message(role="assistant", content="answer"),
        ]
    )

    assert [message.role for message in trajectory.messages] == ["user", "assistant"]
    assert [message.content for message in trajectory.messages] == ["prompt", "answer"]
    assert trajectory.token_ids == []


def test_score_stubbed_messages_uses_stub_trajectory() -> None:
    judge = _RecordingJudge(Reward(total=3.0, components={"elpd": 3.0}, info={"source": "judge"}))

    reward = score_stubbed_messages(
        judge=judge,
        task_name="dummy",
        messages=[{"role": "assistant", "content": "model"}],
        artifacts={"program": "model"},
    )

    assert reward.total == 3.0
    assert judge.calls == [
        (
            "dummy",
            [Message(role="assistant", content="model")],
            {"program": "model"},
        )
    ]


def test_score_stubbed_messages_enriches_runtime_artifacts() -> None:
    judge = _RecordingJudge(Reward(total=1.0, components={"elpd": 1.0}, info={"source": "judge"}))

    score_stubbed_messages(
        judge=judge,
        task_name="dummy",
        messages=[{"role": "assistant", "content": "model"}],
        artifacts={
            "program": "model",
            "shared_trial_record": {
                "trial_id": "completion-001",
                "runtime": {"name": "haskell", "version": "1.0.0"},
            },
            "lazyppl_version": "1.0.0",
        },
    )

    _, _, artifacts = judge.calls[0]
    assert artifacts["language"] == "lazyppl"
    assert artifacts["expected_score_artifact_key"] == "log_lik_matrix"


def test_score_stubbed_messages_total_records_details_and_applies_scale() -> None:
    judge = _RecordingJudge(
        Reward(
            total=2.5,
            components={"elpd": 2.5},
            info={"source": "judge"},
        )
    )
    artifacts: dict[str, object] = {"program": "x = 1"}

    total = score_stubbed_messages_total(
        judge=judge,
        task_name="dummy",
        messages=[{"role": "assistant", "content": "model"}],
        artifacts=artifacts,
        reward_floor=0.1,
        reward_scale=2.0,
    )

    assert total == pytest.approx(5.0)
    assert artifacts["reward_components"] == {"elpd": 2.5}
    assert artifacts["reward_info"] == {
        "source": "judge",
        "reward_source": "judge",
        "evidence_kind": "heuristic",
        "truth_owner": "heuristic",
        "raw_total": 2.5,
        "reward_scale": 2.0,
    }


def test_score_stubbed_messages_reward_preserves_scaled_metadata() -> None:
    judge = _RecordingJudge(
        Reward(
            total=1.5,
            components={"elpd": 1.5},
            info={"source": "judge"},
        )
    )

    reward = score_stubbed_messages_reward(
        judge=judge,
        task_name="dummy",
        messages=[{"role": "assistant", "content": "model"}],
        artifacts={"program": "x = 1"},
        reward_floor=0.1,
        reward_scale=3.0,
    )

    assert reward.total == pytest.approx(4.5)
    assert reward.components == {"elpd": 1.5}
    assert reward.info == {
        "source": "judge",
        "reward_source": "judge",
        "evidence_kind": "heuristic",
        "truth_owner": "heuristic",
        "raw_total": 1.5,
        "reward_scale": 3.0,
    }


def test_record_reward_details_updates_artifacts_and_returns_reward() -> None:
    artifacts: dict[str, object] = {}
    reward = Reward(
        total=1.0,
        components={"elpd": 1.0},
        info={"source": "judge"},
    )

    recorded = record_reward_details(artifacts, reward)

    assert recorded is reward
    assert artifacts == {
        "reward_components": {"elpd": 1.0},
        "reward_info": {
            "source": "judge",
            "reward_source": "judge",
            "evidence_kind": "heuristic",
            "truth_owner": "heuristic",
        },
    }


def test_score_stubbed_messages_reward_normalizes_verifier_validation_source() -> None:
    judge = _RecordingJudge(
        Reward(
            total=-0.123,
            components={"semantic_score": -0.123},
            info={"source": "validation.semantic_score"},
        )
    )

    reward = score_stubbed_messages_reward(
        judge=judge,
        task_name="dummy",
        messages=[{"role": "assistant", "content": "model"}],
        artifacts={"program": "x = 1"},
        reward_floor=-1.0,
    )

    assert reward.info == {
        "source": "validation.semantic_score",
        "reward_source": "validation.semantic_score",
        "evidence_kind": "verifier",
        "truth_owner": "verifier",
        "raw_total": -0.123,
    }


def test_score_stubbed_messages_total_returns_floor_on_judge_failure(caplog) -> None:
    class _ExplodingJudge:
        def score(self, *, task_name, trajectory, artifacts):
            raise ValueError("boom")

    artifacts: dict[str, object] = {}

    with caplog.at_level(logging.WARNING):
        total = score_stubbed_messages_total(
            judge=_ExplodingJudge(),
            task_name="dummy",
            messages=[{"role": "assistant", "content": "model"}],
            artifacts=artifacts,
            reward_floor=0.25,
            logger=logging.getLogger("synthstats.tests"),
            failure_log_prefix="Judge scoring failed in test",
        )

    assert total == pytest.approx(0.25)
    assert artifacts["reward_info"] == {
        "source": "judge_error",
        "reward_source": "judge_error",
        "evidence_kind": "heuristic",
        "truth_owner": "heuristic",
        "judge_error": "boom",
        "reward_floor_applied": True,
    }
    assert any("Judge scoring failed in test" in record.message for record in caplog.records)


class TestCanonicalizeRewardInfo:
    def test_none_returns_annotated(self) -> None:
        from synthstats.core.judge_runtime import _canonicalize_reward_info

        result = _canonicalize_reward_info(None)
        assert result["reward_source"] == "unknown"
        assert result["evidence_kind"] == "heuristic"
        assert result["truth_owner"] == "heuristic"

    def test_empty_dict_returns_annotated(self) -> None:
        from synthstats.core.judge_runtime import _canonicalize_reward_info

        result = _canonicalize_reward_info({})
        assert result["reward_source"] == "unknown"
        assert result["evidence_kind"] == "heuristic"
        assert result["truth_owner"] == "heuristic"

    def test_custom_payload_preserves_and_annotates(self) -> None:
        from synthstats.core.judge_runtime import _canonicalize_reward_info

        result = _canonicalize_reward_info({"custom": 1})
        assert result["custom"] == 1
        assert result["reward_source"] == "unknown"
        assert result["evidence_kind"] == "heuristic"

    def test_verifier_source_classified_correctly(self) -> None:
        from synthstats.core.judge_runtime import _canonicalize_reward_info

        result = _canonicalize_reward_info({"source": "validation.elpd"})
        assert result["evidence_kind"] == "verifier"
        assert result["truth_owner"] == "verifier"
