"""Helpers for scoring message-only trajectories with a Judge."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from synthstats.core.judge import Judge
from synthstats.core.reward_semantics import annotate_reward_info
from synthstats.core.runtime_registry import (
    runtime_name_from_payload,
    score_artifact_key_from_payload,
)
from synthstats.core.types import Message, Reward, Trajectory

type MessageLike = Message | Mapping[str, Any]
_JUDGE_SCORING_ERRORS = (ValueError, TypeError, RuntimeError, ArithmeticError, KeyError)


def build_stub_trajectory(messages: Sequence[MessageLike]) -> Trajectory:
    """Build a trajectory stub from Message objects or message dicts."""
    normalized_messages: list[Message] = []
    for message in messages:
        if isinstance(message, Message):
            normalized_messages.append(message)
        else:
            normalized_messages.append(Message.from_dict(dict(message)))
    return Trajectory.empty_stub(normalized_messages)


def _enrich_runtime_artifacts(artifacts: dict[str, Any]) -> dict[str, Any]:
    runtime_name = runtime_name_from_payload(artifacts)
    if runtime_name is not None:
        artifacts.setdefault("language", runtime_name)
    score_artifact_key = score_artifact_key_from_payload(artifacts)
    if score_artifact_key is not None:
        artifacts.setdefault("expected_score_artifact_key", score_artifact_key)
    return artifacts


def _canonicalize_reward_info(info: Mapping[str, Any] | None) -> dict[str, Any]:
    """Canonicalize reward info with explicit provenance semantics."""
    if not isinstance(info, Mapping):
        return annotate_reward_info({})
    return annotate_reward_info(dict(info))


def score_stubbed_messages(
    *,
    judge: Judge,
    task_name: str,
    messages: Sequence[MessageLike],
    artifacts: dict[str, Any] | None = None,
) -> Reward:
    """Score a message-only trajectory stub with a Judge."""
    prepared_artifacts = _enrich_runtime_artifacts(artifacts or {})
    return judge.score(
        task_name=task_name,
        trajectory=build_stub_trajectory(messages),
        artifacts=prepared_artifacts,
    )


def score_stubbed_messages_reward(
    *,
    judge: Judge,
    task_name: str,
    messages: Sequence[MessageLike],
    artifacts: dict[str, Any],
    reward_floor: float,
    reward_scale: float = 1.0,
    logger: logging.Logger | None = None,
    failure_log_prefix: str = "Judge scoring failed",
) -> Reward:
    """Score a message-only trajectory stub and return a floored/scaled ``Reward``."""
    try:
        reward_obj = score_stubbed_messages(
            judge=judge,
            task_name=task_name,
            messages=messages,
            artifacts=artifacts,
        )
    except _JUDGE_SCORING_ERRORS as exc:
        if logger is not None:
            logger.warning("%s (%s): %s", failure_log_prefix, type(exc).__name__, exc)
        failure_info = _canonicalize_reward_info({"source": "judge_error"})
        failure_info["judge_error"] = str(exc)
        failure_info["reward_floor_applied"] = True
        return Reward(
            total=float(reward_floor),
            components={},
            info=failure_info,
        )

    raw_total = float(reward_obj.total)
    scaled_total = max(raw_total * reward_scale, float(reward_floor))
    info = _canonicalize_reward_info(reward_obj.info)
    info.setdefault("raw_total", raw_total)
    if reward_scale != 1.0:
        info["reward_scale"] = reward_scale
    if scaled_total > raw_total * reward_scale:
        info["reward_floor_applied"] = True

    return Reward(
        total=scaled_total,
        components=dict(reward_obj.components),
        info=info,
        factors=dict(reward_obj.factors),
        scalarization=reward_obj.scalarization,
    )


def record_reward_details(artifacts: dict[str, Any], reward_obj: Reward) -> Reward:
    """Store canonical reward metadata on the artifacts dict and return the reward."""
    artifacts["reward_components"] = reward_obj.components
    artifacts["reward_info"] = _canonicalize_reward_info(reward_obj.info)
    return reward_obj


def score_stubbed_messages_total(
    *,
    judge: Judge,
    task_name: str,
    messages: Sequence[MessageLike],
    artifacts: dict[str, Any],
    reward_floor: float,
    reward_scale: float = 1.0,
    logger: logging.Logger | None = None,
    failure_log_prefix: str = "Judge scoring failed",
) -> float:
    """Score a message-only trajectory stub and return a floored scalar reward."""
    reward_obj = score_stubbed_messages_reward(
        judge=judge,
        task_name=task_name,
        messages=messages,
        artifacts=artifacts,
        reward_floor=reward_floor,
        reward_scale=reward_scale,
        logger=logger,
        failure_log_prefix=failure_log_prefix,
    )
    return record_reward_details(artifacts, reward_obj).total
