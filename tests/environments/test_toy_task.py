from __future__ import annotations

from synthstats.core.types import Message, StepResult
from tests.fixtures import ToyState, ToyTask


class TestToyState:
    def test_default_values(self):
        state = ToyState()
        assert state.step == 0
        assert state.done is False

    def test_custom_values(self):
        state = ToyState(step=3, done=True)
        assert state.step == 3
        assert state.done is True


class TestToyTaskReset:
    def test_reset_returns_initial_state(self):
        task = ToyTask()
        state = task.reset()
        assert isinstance(state, ToyState)
        assert state.step == 0
        assert state.done is False

    def test_reset_with_seed_is_deterministic(self):
        task = ToyTask()
        s1 = task.reset(seed=42)
        s2 = task.reset(seed=42)
        assert s1 == s2

    def test_reset_without_seed(self):
        task = ToyTask()
        state = task.reset(seed=None)
        assert state.step == 0
        assert state.done is False


class TestToyTaskObserve:
    def test_observe_returns_two_messages(self):
        task = ToyTask()
        state = task.reset()
        messages = task.observe(state)
        assert len(messages) == 2

    def test_observe_first_message_is_system(self):
        task = ToyTask()
        state = task.reset()
        messages = task.observe(state)
        assert messages[0].role == "system"
        assert isinstance(messages[0], Message)

    def test_observe_second_message_is_user(self):
        task = ToyTask()
        state = task.reset()
        messages = task.observe(state)
        assert messages[1].role == "user"
        assert "JSON" in messages[1].content


class TestToyTaskStep:
    def test_step_returns_step_result(self):
        task = ToyTask()
        state = task.reset()
        result = task.step(state, action=None)
        assert isinstance(result, StepResult)

    def test_step_always_terminates(self):
        task = ToyTask()
        state = task.reset()
        result = task.step(state, action="anything")
        assert result.done is True

    def test_step_increments_step_counter(self):
        task = ToyTask()
        state = task.reset()
        result = task.step(state, action=None)
        assert result.next_state.step == 1

    def test_step_marks_next_state_as_done(self):
        task = ToyTask()
        state = task.reset()
        result = task.step(state, action=None)
        assert result.next_state.done is True

    def test_step_artifacts_contain_completed(self):
        task = ToyTask()
        state = task.reset()
        result = task.step(state, action=None)
        assert result.artifacts["completed"] is True

    def test_step_does_not_mutate_original_state(self):
        task = ToyTask()
        state = task.reset()
        task.step(state, action=None)
        assert state.step == 0
        assert state.done is False

    def test_step_accepts_arbitrary_action(self):
        task = ToyTask()
        state = task.reset()
        result = task.step(state, action={"type": "submit_program", "code": "x=1"})
        assert result.done is True


class TestToyTaskAttributes:
    def test_name_is_toy(self):
        task = ToyTask()
        assert task.name == "toy"
