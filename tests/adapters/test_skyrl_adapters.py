import pytest
import torch

from synthstats.core.types import Message, Reward, Trajectory, Transition
from synthstats.train.data.adapters import (
    collected_trajectory_to_transitions,
    trajectory_to_transitions,
)
from synthstats.train.data.collectors import CollectedTrajectory


class TestTransitionDataclass:
    def test_defaults(self):
        t = Transition(input_ids=[1, 2], output_ids=[3, 4], logprobs=[-0.1, -0.2])
        assert t.reward == 0.0
        assert t.done is False
        assert t.info == {}

    def test_terminal(self):
        t = Transition(input_ids=[1], output_ids=[2], logprobs=[-0.5], reward=1.5, done=True)
        assert t.reward == 1.5
        assert t.done is True


class TestTrajectoryToTransitions:
    def test_empty(self):
        traj = Trajectory(
            messages=[], token_ids=[], token_logprobs=[], loss_mask=[], reward=Reward(0, {}, {})
        )
        assert trajectory_to_transitions(traj) == []

    def test_single_generation(self):
        traj = Trajectory(
            messages=[Message("user", "hi"), Message("assistant", "hello")],
            token_ids=[[10, 20, 30]],
            token_logprobs=[[-0.1, -0.2, -0.3]],
            loss_mask=[[True, True, True]],
            reward=Reward(2.5, {"elpd": 2.5}, {}),
        )
        transitions = trajectory_to_transitions(traj)
        assert len(transitions) == 1
        assert transitions[0].done is True
        assert transitions[0].reward == 2.5

    def test_multi_generation(self):
        traj = Trajectory(
            messages=[
                Message("user", "hi"),
                Message("assistant", "thinking"),
                Message("user", "ok"),
                Message("assistant", "answer"),
            ],
            token_ids=[[10, 20], [30, 40, 50]],
            token_logprobs=[[-0.1, -0.2], [-0.3, -0.4, -0.5]],
            loss_mask=[[True, True], [True, True, True]],
            reward=Reward(3.0, {}, {}),
        )
        transitions = trajectory_to_transitions(traj)
        assert len(transitions) == 2
        assert transitions[0].reward == 0.0
        assert transitions[0].done is False
        assert transitions[1].reward == 3.0
        assert transitions[1].done is True


class TestCollectedTrajectoryToTransitions:
    def test_single_action(self):
        ct = CollectedTrajectory(
            observations=["obs1"],
            actions=[{"type": "answer", "payload": "42"}],
            log_probs=torch.tensor([-0.5]),
            entropy=torch.tensor([0.1]),
            reward=1.0,
        )
        transitions = collected_trajectory_to_transitions(ct)
        assert len(transitions) == 1
        assert transitions[0].done is True
        assert transitions[0].reward == 1.0
        assert transitions[0].logprobs == [pytest.approx(-0.5)]

    def test_multi_action(self):
        ct = CollectedTrajectory(
            observations=["obs1", "obs2", "obs3"],
            actions=[{"type": "query"}, {"type": "query"}, {"type": "answer"}],
            log_probs=torch.tensor([-0.1, -0.2, -0.3]),
            entropy=torch.tensor([0.1, 0.2, 0.3]),
            reward=2.0,
        )
        transitions = collected_trajectory_to_transitions(ct)
        assert len(transitions) == 3
        assert transitions[0].reward == 0.0
        assert transitions[1].reward == 0.0
        assert transitions[2].reward == 2.0
        assert transitions[2].done is True

    def test_preserves_collected_trajectory_artifacts_in_transition_info(self):
        ct = CollectedTrajectory(
            observations=["obs1"],
            actions=[{"type": "answer", "payload": "42"}],
            log_probs=torch.tensor([-0.5]),
            entropy=torch.tensor([0.1]),
            reward=1.0,
            artifacts={"trace_id": "completion-001:validation"},
        )

        transitions = collected_trajectory_to_transitions(ct)

        assert transitions[0].info["trajectory_artifacts"]["trace_id"] == (
            "completion-001:validation"
        )

    def test_empty(self):
        ct = CollectedTrajectory(
            observations=[],
            actions=[],
            log_probs=torch.zeros(0),
            entropy=torch.zeros(0),
            reward=0.0,
        )
        assert collected_trajectory_to_transitions(ct) == []

    def test_type_error(self):
        with pytest.raises(TypeError, match="expected CollectedTrajectory"):
            collected_trajectory_to_transitions("not a trajectory")


class TestEdgeCases:
    def test_mismatched_token_ids_logprobs_raises(self):
        traj = Trajectory(
            messages=[Message("user", "hi"), Message("assistant", "hello")],
            token_ids=[[10, 20], [30, 40]],
            token_logprobs=[[-0.1, -0.2]],  # only 1 entry, but 2 generations
            loss_mask=[[True, True], [True, True]],
            reward=Reward(1.0, {}, {}),
        )
        with pytest.raises(ValueError, match="token_ids has 2.*token_logprobs has 1"):
            trajectory_to_transitions(traj)
