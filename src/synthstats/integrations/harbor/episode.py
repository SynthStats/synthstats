"""Run a token-preserving agent episode inside a Harbor environment."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from synthstats.core.constants import API_TIMEOUT_S
from synthstats.envs.builders import build_env_from_dict
from synthstats.integrations.harbor.agent import (
    build_action_spec,
    normalize_agent_reward_info,
    prime_chat_history,
    write_agent_result_log,
)
from synthstats.integrations.harbor.generator import (
    _normalize_sampling_source,
    _resolve_max_generate_length,
)


class TokenRolloutClient:
    """Render once, then extend exact token prefixes with actions and observations."""

    def __init__(self, client, *, model: str, sampling: dict):
        self.client = client
        self.model = model
        normalized = _normalize_sampling_source(sampling)
        default_max_tokens = _resolve_max_generate_length(normalized)
        for length_key in ("max_generate_length", "max_tokens", "max_new_tokens", "logprobs"):
            normalized.pop(length_key, None)
        self._default_max_tokens = (
            int(default_max_tokens) if default_max_tokens is not None else None
        )
        self.sampling = {key: value for key, value in normalized.items() if value is not None}

    async def tokenize(self, **fields) -> list[int]:
        response = await self.client.post(
            "/tokenize",
            json={"model": self.model, "add_special_tokens": False, **fields},
            timeout=API_TIMEOUT_S,
        )
        response.raise_for_status()
        tokens = response.json().get("tokens")
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int for t in tokens):
            raise ValueError("The inference endpoint must return exact token IDs")
        return tokens

    async def complete(
        self, prefix: list[int], *, max_tokens: int | None = None
    ) -> tuple[str, dict]:
        resolved_max_tokens = max_tokens if max_tokens is not None else self._default_max_tokens
        if resolved_max_tokens is None:
            raise ValueError("max_tokens must be specified for completion requests")
        response = await self.client.post(
            "/v1/completions",
            json={
                **self.sampling,
                "model": self.model,
                "prompt": prefix,
                "max_tokens": int(resolved_max_tokens),
                "logprobs": 1,
                "return_token_ids": True,
                "return_tokens_as_token_ids": True,
            },
            timeout=API_TIMEOUT_S,
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        tokens = choice.get("token_ids")
        probabilities = (choice.get("logprobs") or {}).get("token_logprobs")
        metrics = {
            "prompt_token_ids": list(prefix),
            "completion_token_ids": tokens,
            "logprobs": probabilities,
            "extra": {"stop_reason": choice.get("finish_reason")},
        }
        # Validate the same payload that the trainer will consume.
        from synthstats.integrations.harbor.converter import extract_causal_token_data

        extract_causal_token_data({"steps": [{"source": "agent", "metrics": metrics}]})
        return choice["text"], metrics


async def run_episode(instruction: str, runtime: dict[str, str], output_dir: Path) -> None:
    task_config = json.loads(runtime["TASK_CONFIG"])
    env = build_env_from_dict(task_config)
    history, _ = env.init()
    history = prime_chat_history(
        history, instruction=instruction, action_spec=build_action_spec(env)
    )
    env.chat_history = history
    model = runtime["MODEL_NAME"]
    sampling = json.loads(runtime["SAMPLING_PARAMS"])
    max_output = int(sampling.pop("max_generate_length"))
    max_sequence = int(runtime["MAX_SEQUENCE_LENGTH"])
    endpoint = runtime["LLM_API_BASE"].rstrip("/").removesuffix("/v1")
    steps: list[dict[str, Any]] = [
        {"step_id": index + 1, "source": message["role"], "message": message["content"]}
        for index, message in enumerate(history)
    ]
    program = None
    reward = 0.0
    reward_info = normalize_agent_reward_info({})
    language = "pymc"
    async with httpx.AsyncClient(base_url=endpoint) as client:
        rollout = TokenRolloutClient(client, model=model, sampling=sampling)
        prefix = await rollout.tokenize(messages=history, add_generation_prompt=True)
        newline = await rollout.tokenize(prompt="\n")
        for _turn in range(int(runtime["MAX_TURNS"])):
            available = min(max_output, max_sequence - len(prefix))
            if available <= 0:
                break
            response, metrics = await rollout.complete(prefix, max_tokens=available)
            steps.append(
                {
                    "step_id": len(steps) + 1,
                    "source": "agent",
                    "message": response,
                    "metrics": metrics,
                }
            )
            prefix = prefix + metrics["completion_token_ids"]
            result = env.step(response)
            if result.get("done", False):
                reward = float(result.get("reward", 0.0))
                metadata = result.get("metadata", {})
                program = metadata.get("program")
                language = metadata.get("language", "pymc")
                reward_info = normalize_agent_reward_info(metadata.get("reward_info"))
                break
            observation = env.chat_history[-1]["content"]
            steps.append({"step_id": len(steps) + 1, "source": "user", "message": observation})
            # Render only the new turn; previous sampled tokens are never re-encoded.
            observation_tokens = await rollout.tokenize(
                messages=[{"role": "user", "content": observation}],
                add_generation_prompt=True,
            )
            prefix = prefix + newline + observation_tokens
    if not any(step["source"] == "agent" for step in steps):
        raise ValueError("The context budget leaves no room for a generated action")
    atif = {
        "schema_version": "ATIF-v1.6",
        "session_id": uuid4().hex,
        "agent": {"name": "synthstats", "version": "0.1.0", "model_name": model},
        "steps": steps,
        "extra": json.loads(runtime.get("TASK_PROVENANCE", "{}")),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trajectory.json").write_text(json.dumps(atif))
    write_agent_result_log(
        {
            "instruction": instruction,
            "program": program,
            "language": language,
            "reward": reward,
            "reward_info": reward_info,
            "task_config": task_config,
            "task_provenance": atif["extra"],
            "turns": sum(step["source"] == "agent" for step in steps),
        },
        path=output_dir / "synthstats_result.json",
    )


if __name__ == "__main__":
    asyncio.run(run_episode(os.environ["TASK_INSTRUCTION"], dict(os.environ), Path("/logs/agent")))
