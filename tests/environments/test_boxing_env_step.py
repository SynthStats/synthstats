from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from synthstats.core.executor import ToolResult
from synthstats.core.types import Message, Program, Reward, StepResult, ToolCall, Trajectory
from synthstats.envs.task_runtime_env import TaskRuntimeEnv, TaskRuntimeEnvConfig
from synthstats.runtime.codecs import ParseError


@dataclass
class _State:
    turn: int = 0


class _LoopTask:
    name = "loop-task"

    def __init__(self) -> None:
        self.actions: list[Any] = []

    def reset(self, seed: int | None = None) -> _State:
        return _State()

    def observe(self, state: _State) -> list[Message]:
        if state.turn == 0:
            return [
                Message(role="system", content="system prompt"),
                Message(role="user", content="initial observation"),
            ]
        return [Message(role="user", content=f"observation-{state.turn}")]

    def step(self, state: _State, action: Any) -> StepResult:
        self.actions.append(action)
        if isinstance(action, Program):
            return StepResult(
                next_state=_State(turn=state.turn + 1),
                done=True,
                artifacts={"program": action.code, "elpd": 2.0},
            )
        return StepResult(
            next_state=_State(turn=state.turn + 1),
            done=False,
            artifacts={"last_action_type": type(action).__name__},
        )

    def native_actions(self) -> frozenset[str]:
        return frozenset({"native_tool"})


class _Codec:
    def parse(self, assistant_text: str) -> Any:
        if assistant_text == "external":
            return ToolCall(
                name="external_tool",
                input={"query": "ages"},
                raw=assistant_text,
            )
        if assistant_text == "native":
            return ToolCall(
                name="native_tool",
                input={"query": "ages"},
                raw=assistant_text,
            )
        if assistant_text == "submit":
            return Program(code="x = 1", language="pymc")
        raise ParseError("bad action")


class _RecordingExecutor:
    name = "external_tool"

    def __init__(self) -> None:
        self.calls: list[tuple[ToolCall, float]] = []

    def execute(self, payload: ToolCall, *, timeout_s: float) -> ToolResult:
        self.calls.append((payload, timeout_s))
        return ToolResult(output="tool-result", success=True)


class _Judge:
    def score(
        self,
        *,
        task_name: str,
        trajectory: Trajectory,
        artifacts: dict[str, Any],
    ) -> Reward:
        assert task_name == "loop-task"
        assert artifacts["program"] == "x = 1"
        return Reward(
            total=1.25,
            components={"base": 1.25},
            info={"source": "judge"},
        )


def _make_env(
    *,
    judge: _Judge | None = None,
    executor: _RecordingExecutor | None = None,
) -> TaskRuntimeEnv:
    return TaskRuntimeEnv(
        task=_LoopTask(),
        codec=_Codec(),
        executors={} if executor is None else {"external_tool": executor},
        judge=judge,
        config=TaskRuntimeEnvConfig(reward_floor=0.1, reward_scale=2.0),
    )


def test_step_executes_external_tools_and_appends_tool_message() -> None:
    executor = _RecordingExecutor()
    env = _make_env(executor=executor)

    env.init()
    result = env.step("external")

    assert result["done"] is False
    assert result["reward"] == 0.0
    assert result["observations"] == [{"role": "user", "content": "observation-1"}]
    assert set(result) == {
        "observations",
        "reward",
        "done",
        "metadata",
        "postprocessed_action",
    }
    assert result["postprocessed_action"] is None
    assert len(executor.calls) == 1
    tool_call, timeout_s = executor.calls[0]
    assert tool_call.name == "external_tool"
    assert timeout_s > 0
    assert env.chat_history[-3:] == [
        {"role": "assistant", "content": "external"},
        {"role": "tool", "content": "tool-result"},
        {"role": "user", "content": "observation-1"},
    ]


def test_step_skips_executor_for_native_tools() -> None:
    executor = _RecordingExecutor()
    env = _make_env(executor=executor)

    env.init()
    result = env.step("native")

    assert result["done"] is False
    assert result["observations"] == [{"role": "user", "content": "observation-1"}]
    assert result["postprocessed_action"] is None
    assert executor.calls == []


def test_step_surfaces_unknown_external_tools_consistently() -> None:
    env = _make_env()

    env.init()
    result = env.step("external")

    assert result["done"] is False
    assert result["postprocessed_action"] is None
    assert env._artifacts["tool_error"] == "Unknown tool: external_tool"
    assert env.chat_history[-3:] == [
        {"role": "assistant", "content": "external"},
        {
            "role": "tool",
            "content": "Error: Unknown tool 'external_tool'. Available tools: []",
        },
        {"role": "user", "content": "observation-1"},
    ]


def test_step_terminal_program_returns_scaled_reward_and_metadata() -> None:
    env = _make_env(judge=_Judge())

    env.init()
    result = env.step("submit")

    assert result["done"] is True
    assert result["reward"] == 2.5
    assert result["metadata"]["program"] == "x = 1"
    assert result["metadata"]["elpd"] == 2.0
    assert result["metadata"]["reward_components"] == {"base": 1.25}
    assert result["metadata"]["reward_info"]["source"] == "judge"
    assert result["metadata"]["reward_info"]["reward_source"] == "judge"
    assert result["metadata"]["reward_info"]["evidence_kind"] == "heuristic"
    assert result["metadata"]["reward_info"]["truth_owner"] == "heuristic"
    assert result["metadata"]["reward_info"]["raw_total"] == 1.25
    assert result["metadata"]["reward_info"]["reward_scale"] == 2.0
    assert result["postprocessed_action"] is None
