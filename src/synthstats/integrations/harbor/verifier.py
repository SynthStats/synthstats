"""Shared Harbor verifier entrypoint for SynthStats task packages."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from synthstats.core.types import Reward
from synthstats.integrations.harbor.reward_artifacts import (
    build_reward_artifact,
    write_reward_artifacts,
)

logger = logging.getLogger(__name__)

DEFAULT_AGENT_RESULT_PATH = Path("/logs/agent/synthstats_result.json")
DEFAULT_VERIFIER_OUTPUT_DIR = Path("/logs/verifier")


def _artifact_reward_source(reward: Reward) -> str:
    reward_info = reward.info if isinstance(reward.info, dict) else {}
    reward_source = reward_info.get("reward_source")
    if isinstance(reward_source, str) and reward_source.strip():
        return reward_source
    source = reward_info.get("source")
    if isinstance(source, str) and source.strip():
        return source
    return "verifier"


def _build_provenance(
    *,
    agent_result: dict[str, Any],
    task_config: dict[str, Any],
    language: str,
) -> dict[str, Any]:
    env_cfg = task_config.get("env")
    env_cfg = env_cfg if isinstance(env_cfg, dict) else {}
    judge_cfg = task_config.get("judge")
    judge_cfg = judge_cfg if isinstance(judge_cfg, dict) else {}

    task_name = env_cfg.get("name")
    provenance = {
        "language": language,
        "task_name": task_name,
        "agent_reward": agent_result.get("reward"),
    }

    env_target = env_cfg.get("_target_")
    if isinstance(env_target, str) and env_target:
        provenance["env_target"] = env_target

    judge_target = judge_cfg.get("_target_")
    if isinstance(judge_target, str) and judge_target:
        provenance["judge_target"] = judge_target

    task_ref = agent_result.get("task_provenance")
    if isinstance(task_ref, dict) and task_ref:
        provenance["task_ref"] = dict(task_ref)

    return provenance


def load_agent_result(
    path: str | Path = DEFAULT_AGENT_RESULT_PATH,
) -> dict[str, Any]:
    """Load the agent result payload emitted inside the Harbor trial."""
    result_path = Path(path)
    with result_path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(
            f"agent result must be a JSON object, got {type(payload).__name__}",
        )
    return payload


def verify_agent_result(agent_result: dict[str, Any]) -> dict[str, Any]:
    """Recompute verifier-owned reward artifacts from the agent payload."""
    task_config = agent_result.get("task_config")
    if not isinstance(task_config, dict) or not task_config:
        raise ValueError("agent result is missing task_config for verifier replay")

    language = str(agent_result.get("language", "pymc"))
    provenance = _build_provenance(
        agent_result=agent_result,
        task_config=task_config,
        language=language,
    )

    program = agent_result.get("program")
    if not isinstance(program, str) or not program.strip():
        reward = Reward(
            total=0.0,
            components={},
            info={
                "loo_status": "not_computed",
                "diagnostics": {},
                "reason": "missing_program",
            },
        )
        return build_reward_artifact(
            reward,
            reward_source="verifier_missing_program",
            provenance=provenance,
        )

    from synthstats.envs.builders import build_env_from_dict

    env = build_env_from_dict(task_config)
    try:
        reward = env.score_program_reward(program, language=language)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    logged_reward = agent_result.get("reward")
    if isinstance(logged_reward, int | float) and float(logged_reward) != reward.total:
        logger.warning(
            "Verifier reward mismatch: agent=%s verifier=%s",
            float(logged_reward),
            reward.total,
        )

    return build_reward_artifact(
        reward,
        reward_source=_artifact_reward_source(reward),
        provenance=provenance,
    )


def run_verifier(
    *,
    agent_result_path: str | Path = DEFAULT_AGENT_RESULT_PATH,
    output_dir: str | Path = DEFAULT_VERIFIER_OUTPUT_DIR,
) -> dict[str, Any]:
    """Load agent output, recompute reward, and persist canonical artifacts."""
    agent_result = load_agent_result(agent_result_path)
    artifact = verify_agent_result(agent_result)
    write_reward_artifacts(artifact, output_dir=output_dir)
    return artifact


def main() -> int:
    run_verifier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
