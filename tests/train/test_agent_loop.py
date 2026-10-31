from __future__ import annotations

from synthstats.core.types import FinalAnswer, Message, Reward, StepResult
from synthstats.train.agent_loop import SynthStatsAgentLoop


class _MockTask:
    name = "test"

    def reset(self, seed=None):
        return {"step": 0}

    def observe(self, state):
        return [Message("user", "describe the data")]

    def step(self, state, action):
        return StepResult(
            next_state={"step": 1},
            done=True,
            artifacts={"reward": 5.0},
        )


class _MockCodec:
    def parse(self, text):
        return FinalAnswer(text=text)


def _mock_generate(messages, temperature=0.7):
    return {
        "text": "my answer",
        "token_ids": [10, 20, 30],
        "prompt_ids": [1, 2],
    }


def test_agent_loop_runs_basic_episode() -> None:
    loop = SynthStatsAgentLoop(
        task=_MockTask(),
        codec=_MockCodec(),
    )

    output = loop.run_episode(_mock_generate, seed=42)

    assert output.reward == 5.0
    assert output.info["done"] is True
    assert output.info["turns"] == 1
    assert output.response_ids == [10, 20, 30]
    assert output.response_mask == [1, 1, 1]


def test_agent_loop_records_judge_reward_details() -> None:
    class MockJudge:
        def score(self, *, task_name, trajectory, artifacts):
            assert trajectory.messages[-1].content == "my answer"
            return Reward(
                total=2.0,
                components={"elpd": 2.0},
                info={"source": "judge"},
            )

    loop = SynthStatsAgentLoop(
        task=_MockTask(),
        codec=_MockCodec(),
        judge=MockJudge(),
    )

    output = loop.run_episode(_mock_generate, seed=42)

    assert output.reward == 2.0
    assert output.info["reward_components"] == {"elpd": 2.0}
    assert output.info["reward_info"] == {
        "source": "judge",
        "reward_source": "judge",
        "evidence_kind": "heuristic",
        "truth_owner": "heuristic",
        "raw_total": 2.0,
    }
