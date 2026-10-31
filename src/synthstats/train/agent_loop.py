"""Multi-turn episode driver for agentic training integrations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from synthstats.core.constants import (
    DEFAULT_TEMPERATURE,
    REWARD_FLOOR_DEFAULT,
    TOOL_TIMEOUT_S_DEFAULT,
)
from synthstats.core.executor import Executor, execute_tool_call
from synthstats.core.judge_runtime import score_stubbed_messages_total
from synthstats.core.task import Task
from synthstats.core.types import ToolCall
from synthstats.runtime.codecs import ActionCodec, ParseError
from synthstats.train.data.types import AgentLoopOutput

if TYPE_CHECKING:
    from synthstats.core.judge import Judge

logger = logging.getLogger(__name__)


class SynthStatsAgentLoop:
    """Drive multi-step agentic episodes over SynthStats tasks."""

    def __init__(
        self,
        task: Task,
        codec: ActionCodec,
        executors: dict[str, Executor] | None = None,
        judge: Judge | None = None,
        *,
        max_turns: int = 20,
        reward_floor: float = REWARD_FLOOR_DEFAULT,
    ) -> None:
        self.task = task
        self.codec = codec
        self.executors: dict[str, Executor] = executors or {}
        self.judge = judge
        self.max_turns = max_turns
        self.reward_floor = reward_floor

    def run_episode(
        self,
        generate_fn: Any,
        *,
        seed: int | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> AgentLoopOutput:
        """Run a single episode. generate_fn(messages, temperature=) -> {text, token_ids, ...}."""
        state = self.task.reset(seed=seed)
        messages = self.task.observe(state)
        chat_history: list[dict[str, str]] = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        prev_obs_content: set[str] = {m.content for m in messages if m.role != "system"}
        task_native = (
            self.task.native_actions() if hasattr(self.task, "native_actions") else frozenset()
        )

        all_response_ids: list[int] = []
        all_response_mask: list[int] = []
        prompt_ids: list[int] = []
        accumulated_artifacts: dict[str, Any] = {}
        turns = 0
        done = False
        tool_calls = 0
        last_observation = ""

        for msg in reversed(messages):
            if msg.role == "user":
                last_observation = msg.content
                break

        while not done and turns < self.max_turns:
            turns += 1

            gen_result = generate_fn(chat_history, temperature=temperature)
            action_text = gen_result["text"] if isinstance(gen_result, dict) else gen_result[0]
            if isinstance(gen_result, dict):
                token_ids = gen_result.get("token_ids", [])
            else:
                token_ids = gen_result[1] if len(gen_result) > 1 else []

            if turns == 1 and isinstance(gen_result, dict):
                prompt_ids = gen_result.get("prompt_ids", [])

            all_response_ids.extend(token_ids)
            all_response_mask.extend([1] * len(token_ids))

            chat_history.append({"role": "assistant", "content": action_text})

            try:
                parsed_action = self.codec.parse(action_text)
            except ParseError as e:
                error_msg = f"Parse error: {e}"
                if last_observation:
                    error_msg = f"{error_msg}\n\n{last_observation}"
                chat_history.append({"role": "user", "content": error_msg})
                continue

            if isinstance(parsed_action, ToolCall) and parsed_action.name not in task_native:
                tool_calls += 1
                tool_result = execute_tool_call(
                    parsed_action,
                    self.executors,
                    timeout_s=TOOL_TIMEOUT_S_DEFAULT,
                    unknown_prefix="Error: ",
                    unknown_available_label="Available tools",
                )
                tool_msg = {"role": "tool", "content": tool_result.output}
                chat_history.append(tool_msg)
                if not tool_result.success:
                    accumulated_artifacts["tool_error"] = tool_result.error

            step_result = self.task.step(state, parsed_action)
            state = step_result.next_state
            done = step_result.done
            accumulated_artifacts.update(step_result.artifacts)

            if not done:
                next_messages = self.task.observe(state)
                for msg in next_messages:
                    if msg.role == "system":
                        continue
                    if msg.content not in prev_obs_content:
                        chat_history.append({"role": msg.role, "content": msg.content})
                        prev_obs_content.add(msg.content)
                        if msg.role == "user":
                            last_observation = msg.content

        reward = self._compute_reward(chat_history, accumulated_artifacts)

        info = {
            "turns": turns,
            "done": done,
            "tool_calls": tool_calls,
            "task_name": self.task.name,
            **{
                key: value
                for key, value in accumulated_artifacts.items()
                if key in ("reward_components", "reward_info", "tool_error")
            },
        }

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=all_response_ids,
            response_mask=all_response_mask,
            reward=reward,
            messages=chat_history,
            info=info,
        )

    def _compute_reward(
        self,
        chat_history: list[dict[str, str]],
        artifacts: dict[str, Any],
    ) -> float:
        """Compute reward via judge if available, else fall back to artifacts."""
        if self.judge is not None:
            return score_stubbed_messages_total(
                judge=self.judge,
                task_name=self.task.name,
                messages=chat_history,
                artifacts=artifacts,
                reward_floor=self.reward_floor,
                logger=logger,
                failure_log_prefix="Judge scoring failed in agent loop",
            )

        if "reward" in artifacts:
            return max(float(artifacts["reward"]), self.reward_floor)
        return self.reward_floor


__all__ = ["SynthStatsAgentLoop", "AgentLoopOutput"]
