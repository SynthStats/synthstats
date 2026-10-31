from typing import Any

from synthstats.core.types import Reward, Trajectory
from synthstats.train.reward import SynthStatsRewardFunction


class RecordingJudge:
    def __init__(self, reward: Reward) -> None:
        self.reward = reward
        self.calls: list[dict[str, object]] = []

    def score(
        self,
        *,
        task_name: str,
        trajectory: Trajectory,
        artifacts: dict[str, Any],
    ) -> Reward:
        self.calls.append(
            {
                "task_name": task_name,
                "messages": [message.to_dict() for message in trajectory.messages],
                "artifacts": dict(artifacts),
            }
        )
        return self.reward


def test_reward_function_returns_scalar_total_and_preserves_inputs() -> None:
    judge = RecordingJudge(
        Reward(
            total=3.5,
            components={"elpd": 3.5},
            info={"samples": 50},
        )
    )
    reward_function = SynthStatsRewardFunction(judge, task_name="dugongs")

    total = reward_function(
        [{"role": "user", "content": "fit the model"}],
        artifacts={"seed": 7},
    )

    assert total == 3.5
    assert judge.calls == [
        {
            "task_name": "dugongs",
            "messages": [{"role": "user", "content": "fit the model"}],
            "artifacts": {"seed": 7},
        }
    ]


def test_reward_function_score_with_details_returns_full_reward() -> None:
    judge = RecordingJudge(
        Reward(
            total=10.0,
            components={"elpd": 10.0},
            info={"samples": 100},
        )
    )
    reward_function = SynthStatsRewardFunction(judge, task_name="test")

    reward = reward_function.score_with_details([{"role": "assistant", "content": "program"}])

    assert reward.total == 10.0
    assert reward.components == {"elpd": 10.0}
    assert reward.info == {"samples": 100}
