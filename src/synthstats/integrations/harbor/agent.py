"""Harbor InstalledAgent for SynthStats statistical modelling tasks.

Runs as a Harbor InstalledAgent during Harbor Trials. Calls the SkyRL HTTP
endpoint (via LLM_API_BASE env var) for LLM inference, interacts with
statistical modelling tasks, and submits programs.

Without the Harbor SDK, SynthStatsInstalledAgent is None and
HARBOR_AGENT_AVAILABLE is False.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from synthstats.core.constants import API_TIMEOUT_S, DEFAULT_TEMPERATURE
from synthstats.core.reward_semantics import annotate_reward_info
from synthstats.core.tool_utils import tool_schemas_to_toolspecs

logger = logging.getLogger(__name__)

# import-guarded Harbor SDK
try:
    from harbor.agents.installed.base import BaseInstalledAgent as _HarborInstalledAgentBase
except ImportError:
    _HarborInstalledAgentBase = None

_BaseInstalledAgent: type[Any] | None = _HarborInstalledAgentBase


def normalize_agent_reward_info(reward_info: Mapping[str, Any] | None) -> dict[str, Any]:
    """Preserve reward metadata while filling canonical truth semantics."""
    if not isinstance(reward_info, Mapping):
        return annotate_reward_info({})
    return annotate_reward_info(dict(reward_info))


def write_agent_result_log(
    result: dict[str, Any],
    *,
    path: str | Path = "/logs/agent/synthstats_result.json",
) -> None:
    """Persist agent results where Harbor task verifiers can read them."""
    payload = dict(result)
    payload.setdefault("language", "pymc")
    task_config = payload.get("task_config")
    payload["task_config"] = dict(task_config) if isinstance(task_config, dict) else {}
    task_provenance = payload.get("task_provenance")
    payload["task_provenance"] = dict(task_provenance) if isinstance(task_provenance, dict) else {}
    payload["reward_info"] = normalize_agent_reward_info(payload.get("reward_info"))

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def require_llm_api_base(env: Mapping[str, str] | None = None) -> str:
    """Require an explicit LLM base URL for Harbor agent runtime."""
    env_map = env if env is not None else os.environ
    value = env_map.get("LLM_API_BASE")
    if value:
        return value
    raise ValueError(
        "LLM_API_BASE must be set explicitly for SynthStats Harbor agents. "
        "Do not rely on an agent-side host.docker.internal fallback."
    )


def build_action_spec(env: Any) -> str:
    """Render the environment's action contract for provider-agnostic prompting."""
    task = getattr(env, "task", None)
    codec = getattr(env, "codec", None)
    if task is None or codec is None or not hasattr(task, "tool_schemas"):
        return ""

    tool_schemas = task.tool_schemas()
    if not tool_schemas:
        return ""

    toolspecs = tool_schemas_to_toolspecs(tool_schemas)
    action_spec = codec.format_action_spec(toolspecs).strip()
    if not action_spec:
        return ""

    return (
        f"{action_spec}\n\nReturn exactly one action per turn. Do not describe the action in prose."
    )


def prime_chat_history(
    chat_history: list[dict[str, str]],
    *,
    instruction: str,
    action_spec: str,
) -> list[dict[str, str]]:
    """Insert Harbor instruction and action contract into the live chat history."""
    primed = [dict(message) for message in chat_history]
    instruction_text = instruction.strip()
    action_spec_text = action_spec.strip()

    if primed and primed[0].get("role") == "system" and action_spec_text:
        system_content = primed[0].get("content", "").rstrip()
        primed[0] = {
            "role": "system",
            "content": "\n\n".join(part for part in (system_content, action_spec_text) if part),
        }
    elif action_spec_text:
        primed.insert(0, {"role": "system", "content": action_spec_text})

    if instruction_text:
        insert_at = 1 if primed and primed[0].get("role") == "system" else 0
        primed.insert(insert_at, {"role": "user", "content": instruction_text})

    return primed


def render_openai_message(message: Mapping[str, Any]) -> str:
    """Convert an OpenAI chat completion message into codec-friendly text."""
    rendered_parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        rendered_parts.append(content)

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls[:1]:
            if not isinstance(tool_call, Mapping):
                continue
            function = tool_call.get("function", {})
            if not isinstance(function, Mapping):
                continue
            name = function.get("name")
            arguments = function.get("arguments", {})
            if not name:
                continue
            payload = {
                "name": name,
                "arguments": arguments,
            }
            tool_call_id = tool_call.get("id")
            if tool_call_id:
                payload["tool_call_id"] = tool_call_id
            rendered_parts.append(f"<tool_call>{json.dumps(payload)}</tool_call>")

    return "\n".join(part for part in rendered_parts if part)


def _make_agent_class():
    """Build SynthStatsInstalledAgent dynamically."""
    if _BaseInstalledAgent is None:
        return None
    base_installed_agent = cast(type[Any], _BaseInstalledAgent)

    class SynthStatsInstalledAgent(base_installed_agent):  # type: ignore[valid-type,misc]
        """Statistical-modelling agent that runs inside a Harbor Docker sandbox.

        Reads task configuration from TASK_CONFIG env var, calls the LLM
        via LLM_API_BASE for reasoning and program generation, and interacts
        with the task environment.

        The agent's execution produces ATIF-compatible output via Harbor's
        trajectory recording. The agent itself focuses on the task loop;
        Harbor handles ATIF serialization via the AgentContext.
        """

        @staticmethod
        def name() -> str:
            return "synthstats"

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            extra_env = getattr(self, "_extra_env", {})
            self._explicit_env = dict(extra_env) if isinstance(extra_env, dict) else {}
            runtime_env = dict(os.environ)
            runtime_env.update(self._explicit_env)
            self._runtime_env = runtime_env
            self.llm_base_url = require_llm_api_base(self._runtime_env)
            task_config_str = self._runtime_env.get("TASK_CONFIG", "{}")
            try:
                self.task_config = json.loads(task_config_str)
            except json.JSONDecodeError:
                self.task_config = {}
            task_provenance_str = self._runtime_env.get("TASK_PROVENANCE", "{}")
            try:
                self.task_provenance = json.loads(task_provenance_str)
            except json.JSONDecodeError:
                self.task_provenance = {}
            self.max_turns = int(self._runtime_env.get("MAX_TURNS", "20"))

        async def install(self, environment: Any) -> None:
            """Validate the runtime and prepare Harbor's mounted log directory."""
            await self.exec_as_root(
                environment,
                command="mkdir -p /logs/agent && chmod 777 /logs/agent",
            )
            await self.exec_as_agent(
                environment,
                command="python --version",
            )

        async def run(self, instruction: str, environment: Any, context: Any) -> None:
            """Run the complete episode in Harbor's supplied environment."""
            keys = (
                "TASK_CONFIG",
                "LLM_API_BASE",
                "MODEL_NAME",
                "SAMPLING_PARAMS",
                "MAX_SEQUENCE_LENGTH",
                "MAX_TURNS",
            )
            missing = [key for key in keys if key not in self._runtime_env]
            if missing:
                raise ValueError(f"Missing Harbor episode settings: {', '.join(missing)}")
            runtime = {key: self._runtime_env[key] for key in keys}
            explicit_pythonpath = self._explicit_env.get("PYTHONPATH")
            runtime["PYTHONPATH"] = (
                explicit_pythonpath
                if isinstance(explicit_pythonpath, str) and explicit_pythonpath
                else "/workspace/synthstats/src:/workspace/synthstats"
            )
            runtime["TASK_PROVENANCE"] = json.dumps(self.task_provenance)
            runtime["TASK_INSTRUCTION"] = instruction
            result = await self.exec_as_agent(
                environment,
                command="python -m synthstats.integrations.harbor.episode",
                env=runtime,
            )
            if result.return_code != 0:
                raise RuntimeError(f"Harbor episode failed: {result.stderr}")

        def populate_context_post_run(self, context: Any) -> None:
            """Backfill context from the emitted agent payload when Harbor needs it."""
            result_path = self.logs_dir / "synthstats_result.json"
            if not result_path.exists():
                return

            try:
                payload = json.loads(result_path.read_text())
            except json.JSONDecodeError:
                logger.warning("Failed to parse agent result log at %s", result_path)
                return

            existing_metadata = getattr(context, "metadata", None)
            merged_metadata = dict(existing_metadata) if isinstance(existing_metadata, dict) else {}
            merged_metadata.update(payload)
            context.metadata = merged_metadata

        def _build_env(self) -> Any:
            """Build the task runtime environment from task config."""
            from synthstats.envs.builders import build_env_from_dict

            return build_env_from_dict(self.task_config)

        async def _call_llm(
            self,
            client: Any,
            messages: list[dict[str, str]],
            *,
            tool_schemas: list[dict[str, Any]] | None = None,
        ) -> str:
            """Call the SkyRL HTTP endpoint for LLM inference."""
            _max_tokens_env = self._runtime_env.get("MAX_OUTPUT_TOKENS")
            payload: dict = {
                "model": "default",
                "messages": messages,
                "temperature": DEFAULT_TEMPERATURE,
            }
            if _max_tokens_env is not None:
                payload["max_tokens"] = int(_max_tokens_env)
            if tool_schemas:
                payload["tools"] = tool_schemas
                payload["tool_choice"] = "required"
                payload["parallel_tool_calls"] = False
            response = await client.post(
                "/chat/completions",
                json=payload,
                timeout=API_TIMEOUT_S,
            )
            response.raise_for_status()
            data = response.json()
            message = data["choices"][0]["message"]
            rendered = render_openai_message(message)
            if rendered:
                return rendered
            return message.get("content", "") or ""

        def _get_tool_schemas(self, env: Any) -> list[dict[str, Any]] | None:
            task = getattr(env, "task", None)
            if task is None or not hasattr(task, "tool_schemas"):
                return None
            tool_schemas = task.tool_schemas()
            return tool_schemas or None

    return SynthStatsInstalledAgent


SynthStatsInstalledAgent = _make_agent_class()

HARBOR_AGENT_AVAILABLE = SynthStatsInstalledAgent is not None

__all__ = [
    "SynthStatsInstalledAgent",
    "HARBOR_AGENT_AVAILABLE",
    "build_action_spec",
    "prime_chat_history",
    "render_openai_message",
    "write_agent_result_log",
]
