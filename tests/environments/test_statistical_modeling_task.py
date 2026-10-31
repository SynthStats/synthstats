from __future__ import annotations

from synthstats.tasks.statistical_modeling.task import (
    StatisticalModelingState,
    StatisticalModelingTask,
)


def test_statistical_modeling_task_exports() -> None:
    assert StatisticalModelingTask.__name__ == "StatisticalModelingTask"
    assert StatisticalModelingState.__name__ == "StatisticalModelingState"


def test_statistical_modeling_task_name_and_state() -> None:
    task = StatisticalModelingTask(env_name="dugongs", max_steps=10)

    assert task.name == "statistical_modeling"

    state = task.reset(seed=42)
    assert type(state) is StatisticalModelingState
    assert state.observations == []
    assert state.step == 0
    assert state.done is False
