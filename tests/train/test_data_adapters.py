from __future__ import annotations

import types

import torch

from synthstats.core.types import Message, Reward, Trajectory
from synthstats.train.data import adapters as data_adapters
from synthstats.train.data.collectors import CollectedTrajectory


class TestDataAdapters:
    def test_trajectory_to_transitions(self):
        traj = Trajectory(
            messages=[Message("user", "hi"), Message("assistant", "hello")],
            token_ids=[[10, 20, 30]],
            token_logprobs=[[-0.1, -0.2, -0.3]],
            loss_mask=[[True, True, True]],
            reward=Reward(2.5, {"elpd": 2.5}, {}),
        )

        transitions = data_adapters.trajectory_to_transitions(traj)

        assert len(transitions) == 1
        assert transitions[0].done is True
        assert transitions[0].reward == 2.5

    def test_collected_trajectory_to_transitions_preserves_artifacts(self):
        traj = CollectedTrajectory(
            observations=["obs1"],
            actions=[{"type": "answer", "payload": "42"}],
            log_probs=torch.tensor([-0.5]),
            entropy=torch.tensor([0.1]),
            reward=1.0,
            artifacts={"trace_id": "completion-001:validation"},
        )

        transitions = data_adapters.collected_trajectory_to_transitions(traj)

        assert transitions[0].info["trajectory_artifacts"]["trace_id"] == (
            "completion-001:validation"
        )

    def test_batch_agent_loop_outputs(self):
        outputs = [
            types.SimpleNamespace(response_ids=[1, 2], response_mask=[1, 1], reward=0.5),
            types.SimpleNamespace(response_ids=[3], response_mask=[1], reward=1.5),
        ]

        batch = data_adapters.batch_agent_loop_outputs(outputs, pad_id=0)

        assert batch["response_ids"].tolist() == [[1, 2], [3, 0]]
        assert batch["response_mask"].tolist() == [[1.0, 1.0], [1.0, 0.0]]
        assert batch["rewards"].tolist() == [0.5, 1.5]
