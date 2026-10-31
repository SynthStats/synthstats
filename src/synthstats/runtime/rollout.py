"""Rollout loop: runs episodes over any task/policy/codec combination."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from synthstats.core.constants import DEFAULT_MAX_TURNS, TOOL_TIMEOUT_S_DEFAULT
from synthstats.core.executor import Executor, execute_tool_call
from synthstats.core.judge import Judge
from synthstats.core.policy import GenConfig, GenerativePolicy
from synthstats.core.task import Task
from synthstats.core.types import (
    Message,
    Reward,
    ToolCall,
    Trajectory,
)
from synthstats.runtime.codecs import ActionCodec, ParseError


@dataclass
class RolloutConfig:
    max_steps: int = DEFAULT_MAX_TURNS
    tool_timeout_s: float = TOOL_TIMEOUT_S_DEFAULT
    seed: int | None = None
    gen_config: GenConfig = field(default_factory=GenConfig)


def rollout_episode(
    task: Task,
    policy: GenerativePolicy,
    codec: ActionCodec,
    executors: dict[str, Executor],
    judge: Judge,
    cfg: RolloutConfig,
) -> Trajectory:
    """Run one episode: reset -> observe -> generate -> parse -> step -> score."""
    state = task.reset(seed=cfg.seed)
    messages: list[Message] = []
    token_ids: list[list[int]] = []
    token_logprobs: list[list[float]] = []
    loss_mask: list[list[bool]] = []
    eos_logprobs: list[list[float]] = []
    accumulated_artifacts: dict[str, Any] = {}
    done = False
    steps = 0

    # initial observation: include system prompt + first user message
    task_native = task.native_actions() if hasattr(task, "native_actions") else frozenset()

    obs_messages = task.observe(state)
    messages.extend(obs_messages)
    prev_obs_content: set[str] = {m.content for m in obs_messages if m.role != "system"}

    while not done and steps < cfg.max_steps:
        generation = policy.generate(messages, gen=cfg.gen_config)

        messages.append(Message(role="assistant", content=generation.text))
        token_ids.append(generation.token_ids)
        token_logprobs.append(generation.token_logprobs)
        loss_mask.append(_build_loss_mask(generation.token_ids))
        eos_logprobs.append(generation.eos_logprobs)

        try:
            action = codec.parse(generation.text)
        except ParseError as e:
            error_msg = f"parse error: {e}"
            messages.append(Message(role="user", content=error_msg))
            steps += 1
            continue

        # Skip task-owned tools; surface unknown executor-backed tools consistently.
        if isinstance(action, ToolCall) and action.name not in task_native:
            tool_result = execute_tool_call(
                action,
                executors,
                timeout_s=cfg.tool_timeout_s,
                unknown_prefix="Error: ",
                unknown_available_label="Available tools",
            )
            messages.append(
                Message(
                    role="tool",
                    content=tool_result.output,
                    tool_call_id=action.name,
                )
            )
            if not tool_result.success:
                accumulated_artifacts["tool_error"] = tool_result.error

        step_result = task.step(state, action)
        state = step_result.next_state
        done = step_result.done
        accumulated_artifacts.update(step_result.artifacts)
        steps += 1

        if not done:
            next_obs = task.observe(state)
            for msg in next_obs:
                if msg.role == "system":
                    continue
                # only add genuinely new observations (skip duplicated content)
                if msg.content not in prev_obs_content:
                    messages.append(msg)
                    prev_obs_content.add(msg.content)

    trajectory = Trajectory(
        messages=messages,
        token_ids=token_ids,
        token_logprobs=token_logprobs,
        loss_mask=loss_mask,
        reward=Reward(total=0.0, components={}, info={}),
        eos_logprobs=eos_logprobs,
    )

    reward = judge.score(
        task_name=task.name,
        trajectory=trajectory,
        artifacts=accumulated_artifacts,
    )

    return replace(trajectory, reward=reward)


def _build_loss_mask(token_ids: list[int]) -> list[bool]:
    """Build per-token loss mask.

    TB/SubTB optimize over the full latent trajectory (Z, Y), so all generated
    tokens (including <think> reasoning) are included in the loss by default.
    """
    return [True] * len(token_ids)
