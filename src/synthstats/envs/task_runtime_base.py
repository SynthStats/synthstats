"""Shared generic implementation for task runtime environments.

This module owns the execution flow for task, codec, executor, and judge
composition; wrappers expose concrete environments without duplicating it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from synthstats.core.constants import (
    REWARD_FLOOR_DEFAULT,
    TOOL_TIMEOUT_S_DEFAULT,
    validate_reward_floor,
)
from synthstats.core.executor import execute_tool_call
from synthstats.core.import_helpers import try_import_symbol
from synthstats.core.judge_runtime import (
    record_reward_details,
    score_stubbed_messages_reward,
)
from synthstats.core.reward_semantics import annotate_reward_info
from synthstats.core.runtime_registry import (
    default_runtime_name,
    get_runtime_spec,
    normalize_runtime_name,
    supported_runtime_names,
)
from synthstats.core.types import Reward

if TYPE_CHECKING:
    from synthstats.core.executor import Executor
    from synthstats.core.judge import Judge
    from synthstats.core.task import Task
    from synthstats.runtime.codecs import ActionCodec


BaseTextEnv = try_import_symbol(
    "BaseTextEnv",
    "skyrl_gym.envs.base_text_env",
)
BaseTextEnvStepOutput = try_import_symbol(
    "BaseTextEnvStepOutput",
    "skyrl_gym.envs.base_text_env",
)
ConversationType = try_import_symbol(
    "ConversationType",
    "skyrl_gym.envs.base_text_env",
)
SKYRL_AVAILABLE = BaseTextEnv is not None
if not SKYRL_AVAILABLE:
    BaseTextEnv = object  # type: ignore[assignment,misc]
    BaseTextEnvStepOutput = dict[str, Any]  # type: ignore[misc,assignment]
    ConversationType = list[dict[str, str]]  # type: ignore[misc,assignment]


logger = logging.getLogger(__name__)
DEFAULT_RUNTIME_NAME = default_runtime_name()
SUPPORTED_RUNTIME_NAMES = frozenset(supported_runtime_names())
ConversationMessage = dict[str, str]
ConversationList = list[ConversationMessage]
StepOutput = dict[str, Any]


@dataclass
class TaskRuntimeConfig:
    max_turns: int = 20
    reward_floor: float = REWARD_FLOOR_DEFAULT
    reward_scale: float = 1.0
    include_format_reward: bool = True
    format_reward_weight: float = 0.1
    evaluation_seed: int | None = 0

    def __post_init__(self) -> None:
        validate_reward_floor(self.reward_floor)


class TaskRuntimeEnvBase(BaseTextEnv):  # type: ignore[misc, valid-type]
    """Generic task runtime environment with integrated Judge."""

    def __init__(
        self,
        task: Task,
        codec: ActionCodec,
        executors: dict[str, Executor] | None = None,
        judge: Judge | None = None,
        config: TaskRuntimeConfig | None = None,
        *,
        name: str | None = None,
        max_steps: int | None = None,
        worker_pool_size: int | None = None,
    ) -> None:
        if hasattr(super(), "__init__"):
            super().__init__()

        self.task = task
        self.codec = codec
        self.executors: dict[str, Executor] = executors or {}
        self.judge = judge
        self.config = config or TaskRuntimeConfig()

        self.max_turns = self.config.max_turns
        self._random_seed = self.config.evaluation_seed
        self._worker_pool_size = worker_pool_size

        self._pool: Any = None
        self.chat_history: ConversationList = []
        self._state: Any = None
        self._turns: int = 0
        self._done: bool = False
        self._trajectory_messages: ConversationList = []
        self._artifacts: dict[str, Any] = {}

    def init(
        self, prompt: ConversationList | None = None
    ) -> tuple[ConversationList, dict[str, Any]]:
        self._turns = 0
        self._done = False
        self._trajectory_messages = []
        self._artifacts = {}

        self._state = self.task.reset(seed=self._random_seed)
        messages = self.task.observe(self._state)

        self.chat_history = list(prompt) if prompt else []
        for msg in messages:
            msg_dict = {"role": msg.role, "content": msg.content}
            self.chat_history.append(msg_dict)
            self._trajectory_messages.append(msg_dict)

        return self.chat_history, {"task_name": self.task.name}

    def step(self, action: str) -> StepOutput:
        if self._done:
            return self._make_output([], 0.0, done=True, metadata={"already_done": True})

        self._turns += 1
        self._append_message("assistant", action)

        if self._turns > self.max_turns:
            return self._terminate(metadata={"max_turns_exceeded": True})

        parsed_action = self._parse_action(action)
        if parsed_action is None:
            return self._last_parse_error_output

        self._execute_non_task_tools(parsed_action)

        step_result = self.task.step(self._state, parsed_action)
        self._state = step_result.next_state
        self._artifacts.update(step_result.artifacts)

        if step_result.done or self._turns >= self.max_turns:
            return self._terminate(metadata=self._artifacts)

        return self._build_observations()

    def _append_message(
        self,
        role: Literal["system", "user", "assistant", "tool"],
        content: str,
    ) -> None:
        msg = {"role": role, "content": content}
        self.chat_history.append(msg)
        self._trajectory_messages.append(msg)

    def _parse_action(self, action: str) -> Any | None:
        from synthstats.runtime.codecs import ParseError

        try:
            return self.codec.parse(action)
        except ParseError as exc:
            error_msg = f"Parse error: {exc}"
            self._append_message("user", error_msg)
            self._last_parse_error_output = self._make_output(
                [{"role": "user", "content": error_msg}],
                0.0,
                done=False,
                metadata={"parse_error": str(exc)},
            )
            return None

    def _execute_non_task_tools(self, parsed_action: Any) -> None:
        from synthstats.core.types import ToolCall

        if not isinstance(parsed_action, ToolCall):
            return
        task_native = (
            self.task.native_actions() if hasattr(self.task, "native_actions") else frozenset()
        )
        if parsed_action.name in task_native:
            return
        result = execute_tool_call(
            parsed_action,
            self.executors,
            timeout_s=TOOL_TIMEOUT_S_DEFAULT,
            unknown_prefix="Error: ",
            unknown_available_label="Available tools",
        )
        self._append_message("tool", result.output)
        if not result.success:
            self._artifacts["tool_error"] = result.error

    def _terminate(self, metadata: dict[str, Any] | None = None) -> StepOutput:
        self._done = True
        reward = self._compute_final_reward()
        return self._make_output([], reward, done=True, metadata=metadata or self._artifacts)

    def _build_observations(self) -> StepOutput:
        next_messages = self.task.observe(self._state)
        new_observations: list[dict[str, str]] = []
        for msg in next_messages:
            if msg.role == "system":
                continue
            obs_dict = {"role": msg.role, "content": msg.content}
            self.chat_history.append(obs_dict)
            self._trajectory_messages.append(obs_dict)
            new_observations.append(obs_dict)
        return self._make_output(new_observations, 0.0, done=False)

    @staticmethod
    def _make_output(
        observations: list[dict[str, str]],
        reward: float,
        *,
        done: bool,
        metadata: dict[str, Any] | None = None,
    ) -> StepOutput:
        return {
            "observations": observations,
            "reward": reward,
            "done": done,
            "metadata": metadata or {},
            "postprocessed_action": None,
        }

    def _run_judge_reward(
        self, trajectory_messages: list[dict[str, str]], artifacts: dict[str, Any]
    ) -> Reward:
        reward = score_stubbed_messages_reward(
            judge=self.judge,  # type: ignore[arg-type]
            task_name=self.task.name,
            messages=trajectory_messages,
            artifacts=artifacts,
            reward_floor=self.config.reward_floor,
            reward_scale=self.config.reward_scale,
            logger=logger,
            failure_log_prefix="Judge scoring failed",
        )
        return record_reward_details(artifacts, reward)

    def _get_pool(self) -> Any:
        if self._pool is not None:
            return self._pool
        if self._worker_pool_size is None:
            return None
        from synthstats.executors.worker_pool import PyMCWorkerPool

        self._pool = PyMCWorkerPool(pool_size=self._worker_pool_size)
        self._pool.start()
        logger.info("task runtime env: started worker pool with %d workers", self._pool.pool_size)
        return self._pool

    def _build_elpd_runner(self) -> Any:
        raise NotImplementedError

    def _ensure_artifacts_computed(self, artifacts: dict[str, Any]) -> None:
        raw_language = artifacts.get("language")
        language = normalize_runtime_name(raw_language, default=DEFAULT_RUNTIME_NAME)
        language = language or str(raw_language or DEFAULT_RUNTIME_NAME)
        program = artifacts.get("program")
        if not program:
            return
        if language not in SUPPORTED_RUNTIME_NAMES:
            logger.warning("unsupported language %r; skipping artifact computation", language)
            return
        runtime_spec = get_runtime_spec(language)
        if (
            runtime_spec.score_artifact_key == "log_lik_matrix"
            and "log_lik_matrix" not in artifacts
        ):
            self._populate_lazyppl_loglik(program, artifacts)
        elif runtime_spec.score_artifact_key == "elpd" and "elpd" not in artifacts:
            result = self._build_elpd_runner().execute(program)
            artifacts.update(result)

    def score_program(self, program: str, language: str = DEFAULT_RUNTIME_NAME) -> float:
        return self.score_program_reward(program, language=language).total

    def score_program_reward(self, program: str, language: str = DEFAULT_RUNTIME_NAME) -> Reward:
        normalized_language = normalize_runtime_name(language, default=DEFAULT_RUNTIME_NAME)
        artifacts: dict[str, Any] = {
            "program": program,
            "language": normalized_language or language,
        }
        return self._finalize_reward_object(artifacts, trajectory_messages=[])

    def _populate_lazyppl_loglik(self, program: str, artifacts: dict[str, Any]) -> None:
        executor = self.executors.get("lazyppl")
        if executor is None:
            logger.warning("no lazyppl executor")
            return

        if not hasattr(executor, "execute_for_loglik"):
            logger.warning("lazyppl executor missing execute_for_loglik")
            return

        env = self.task.env if hasattr(self.task, "env") else None
        columns = (
            env.get_lazyppl_columns()
            if env is not None and hasattr(env, "get_lazyppl_columns")
            else None
        )
        data_rows = (
            env.get_lazyppl_eval_data()
            if env is not None and hasattr(env, "get_lazyppl_eval_data")
            else None
        )

        if columns is None or data_rows is None:
            logger.warning("Environment %s does not provide lazyppl eval data", self.task.name)
            return

        log_lik_matrix = executor.execute_for_loglik(program, columns, data_rows)
        if log_lik_matrix is not None:
            artifacts["log_lik_matrix"] = log_lik_matrix
            logger.debug("log-lik matrix: %s", log_lik_matrix.shape)

    def _compute_final_reward(self) -> float:
        if "program" in self._artifacts:
            logger.debug("[diag] program (first 500 chars): %.500s", self._artifacts["program"])
        return self._finalize_reward_object(
            self._artifacts,
            self._trajectory_messages,
        ).total

    def _reward_floor_object(
        self,
        *,
        source: str,
        reward: float = 0.0,
        extra_info: dict[str, Any] | None = None,
    ) -> Reward:
        scaled_total = max(float(reward), float(self.config.reward_floor))
        info: dict[str, Any] = annotate_reward_info({}, reward_source=source, source=source)
        if scaled_total > float(reward):
            info["reward_floor_applied"] = True
        if extra_info:
            info.update(extra_info)
        components = {"raw": float(reward)} if reward else {}
        return Reward(total=scaled_total, components=components, info=info)

    def _finalize_reward_object(
        self,
        artifacts: dict[str, Any],
        trajectory_messages: list[dict[str, str]],
    ) -> Reward:
        self._ensure_artifacts_computed(artifacts)

        raw_language = artifacts.get("language")
        language = normalize_runtime_name(raw_language, default=DEFAULT_RUNTIME_NAME)
        language = language or str(raw_language or DEFAULT_RUNTIME_NAME)
        if language not in SUPPORTED_RUNTIME_NAMES:
            return self._reward_floor_object(
                source="unsupported_runtime",
                extra_info={"language": language},
            )
        runtime_spec = get_runtime_spec(language)
        expected_artifact_key = runtime_spec.score_artifact_key
        if expected_artifact_key not in artifacts and "program" in artifacts:
            return self._reward_floor_object(
                source="missing_artifacts",
                extra_info={
                    "language": language,
                    "expected_artifact_key": expected_artifact_key,
                },
            )

        if self.judge is None:
            logger.warning("score_program called without a Judge; returning reward_floor")
            reward = artifacts.get("reward", 0.0)
            return self._reward_floor_object(source="no_judge", reward=float(reward))

        final = self._run_judge_reward(trajectory_messages, artifacts)
        logger.debug(
            "[diag] reward=%.4f components=%s",
            final.total,
            artifacts.get("reward_components"),
        )
        return final

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def get_metrics(self) -> dict[str, Any]:
        metrics = {
            "turns": self._turns,
            "done": self._done,
            "task_name": self.task.name,
        }
        if "reward_components" in self._artifacts:
            metrics["reward_components"] = self._artifacts["reward_components"]
        return metrics


__all__ = ["SKYRL_AVAILABLE", "TaskRuntimeConfig", "TaskRuntimeEnvBase"]
