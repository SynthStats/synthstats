"""ATIF <-> SynthStats type converters.

Pure functions converting between Harbor's Agent Trajectory Interchange
Format (ATIF v1.5/v1.6) and SynthStats core types. No Harbor SDK required.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synthstats.core.reward_semantics import annotate_reward_info
from synthstats.core.types import Message, Reward, Trajectory
from synthstats.integrations.harbor.reward_artifacts import reward_artifact_to_reward

logger = logging.getLogger(__name__)

_SUPPORTED_SCHEMA_PREFIXES = ("ATIF-v1.4", "ATIF-v1.5", "ATIF-v1.6")
_SOURCE_TO_ROLE = {"agent": "assistant", "user": "user", "system": "system"}


@dataclass(frozen=True)
class AtifGeneration:
    """One assistant generation extracted from an ATIF trajectory."""

    token_ids: list[int]
    logprobs: list[float]
    loss_mask: list[bool]


def validate_atif_schema(atif: dict[str, Any]) -> tuple[bool, str]:
    """Lightweight schema check. Returns (ok, error_message)."""
    version = atif.get("schema_version", "")
    if not any(version.startswith(p) for p in _SUPPORTED_SCHEMA_PREFIXES):
        return False, f"unsupported schema_version: {version!r}"
    if "steps" not in atif or not isinstance(atif["steps"], list):
        return False, "missing or invalid 'steps' array"
    return True, ""


def atif_to_messages(atif: dict[str, Any]) -> list[Message]:
    """Convert ATIF steps into a flat list of SynthStats Messages."""
    messages: list[Message] = []

    for step in atif.get("steps", []):
        source = step.get("source", "")
        role = _SOURCE_TO_ROLE.get(source, "user")

        # build content: prepend reasoning in <think> tags for agent steps
        reasoning = step.get("reasoning_content")
        raw_message = step.get("message", "")
        # handle v1.6 multimodal content (list of ContentPart)
        if isinstance(raw_message, list):
            for part in raw_message:
                if isinstance(part, dict) and part.get("type", "text") != "text":
                    logger.warning(
                        "dropping non-text content part type=%s in step source=%s",
                        part.get("type"),
                        source,
                    )
            raw_message = " ".join(
                part.get("text", "")
                for part in raw_message
                if isinstance(part, dict) and part.get("type", "text") == "text"
            )

        content = raw_message

        if content or (role == "assistant" and reasoning):
            messages.append(
                Message(
                    role=role,
                    content=content,
                    reasoning_content=reasoning if role == "assistant" else None,
                )
            )

        # flatten tool calls into assistant messages
        for tc in step.get("tool_calls", []):
            tc_content = json.dumps(
                {
                    "name": tc.get("function_name", ""),
                    "arguments": tc.get("arguments", {}),
                },
                ensure_ascii=False,
            )
            messages.append(
                Message(
                    role="assistant",
                    content=tc_content,
                    tool_call_id=tc.get("tool_call_id"),
                )
            )

        # flatten observation results into tool messages
        obs = step.get("observation")
        if obs and isinstance(obs, dict):
            for result in obs.get("results", []):
                obs_content = result.get("content", "")
                if isinstance(obs_content, list):
                    for part in obs_content:
                        if isinstance(part, dict) and part.get("type", "text") != "text":
                            logger.warning(
                                "dropping non-text observation part type=%s",
                                part.get("type"),
                            )
                    obs_content = " ".join(
                        part.get("text", "")
                        for part in obs_content
                        if isinstance(part, dict) and part.get("type", "text") == "text"
                    )
                messages.append(
                    Message(
                        role="tool",
                        content=str(obs_content),
                        tool_call_id=result.get("source_call_id"),
                    )
                )

    return messages


def _is_copied_context_step(step: dict[str, Any]) -> bool:
    if bool(step.get("is_copied_context")):
        return True
    extra = step.get("extra")
    if isinstance(extra, dict) and bool(extra.get("is_copied_context")):
        return True
    metrics = step.get("metrics")
    if isinstance(metrics, dict) and bool(metrics.get("is_copied_context")):
        return True
    return False


def _exact_token_ids(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Agent step requires exact {label} token IDs")
    tokens: list[int] = []
    for token in value:
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise ValueError(f"Agent step requires exact {label} token IDs")
        tokens.append(token)
    return tokens


def validate_training_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a token-level training payload across metadata and ATIF routes."""
    prompt_ids = _exact_token_ids(payload.get("prompt_ids"), "prompt")
    response_ids = _exact_token_ids(payload.get("response_ids"), "completion")

    logprobs = payload.get("logprobs")
    if not isinstance(logprobs, list):
        raise ValueError("Rollout metadata requires behavior log probabilities")

    loss_mask = payload.get("loss_mask")
    if (
        not isinstance(loss_mask, list)
        or len(response_ids) != len(loss_mask)
        or len(response_ids) != len(logprobs)
    ):
        raise ValueError("Rollout token IDs, probabilities and masks must align")

    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value > 0
        for value in logprobs
    ):
        raise ValueError("Behavior log probabilities must be finite and nonpositive")

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1)
        for value in loss_mask
    ):
        raise ValueError("Agent loss mask must align with generated tokens")

    return {
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
        "logprobs": [float(value) for value in logprobs],
        "loss_mask": [int(value) for value in loss_mask],
        "stop_reason": str(payload.get("stop_reason", "completed")),
    }


def extract_agent_generations(
    atif: dict[str, Any],
    *,
    tokenizer: Any | None = None,
    mask_thinking: bool = False,
) -> list[AtifGeneration]:
    """Extract assistant generations from ATIF with optional think masking."""
    generations: list[AtifGeneration] = []
    for step in atif.get("steps", []):
        if step.get("source") != "agent":
            continue
        metrics = step.get("metrics")
        if not metrics:
            continue
        raw_ids = metrics.get("completion_token_ids")
        if not raw_ids:
            continue
        ids = _exact_token_ids(raw_ids, "completion")
        copied = _is_copied_context_step(step)
        lps = metrics.get("logprobs")
        if copied and lps is None:
            lps = [0.0] * len(ids)
        if not isinstance(lps, list) or len(lps) != len(ids):
            raise ValueError(
                "Training generations require behavior log probabilities for every token"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value > 0
            for value in lps
        ):
            raise ValueError("Behavior log probabilities must be finite and nonpositive")
        if copied:
            loss_mask = [False] * len(ids)
        elif mask_thinking:
            reasoning = step.get("reasoning_content")
            message = step.get("message", "")
            if isinstance(message, list):
                message = " ".join(
                    part.get("text", "")
                    for part in message
                    if isinstance(part, dict) and part.get("type", "text") == "text"
                )
            loss_mask = build_loss_mask(
                list(ids),
                reasoning,
                tokenizer,
                message_content=message,
            )
        else:
            loss_mask = [True] * len(ids)
        generations.append(
            AtifGeneration(
                token_ids=list(ids),
                logprobs=[float(value) for value in lps],
                loss_mask=loss_mask,
            )
        )
    return generations


def extract_causal_token_data(atif: dict[str, Any]) -> dict[str, Any] | None:
    """Join exact model-visible prefixes and actions into one causal training sequence."""
    prompt: list[int] | None = None
    sequence: list[int] = []
    masks: list[int] = []
    logprobs: list[float] = []
    for step in atif.get("steps", []):
        if step.get("source") != "agent":
            continue
        metrics = step.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("Agent step lacks causal token metrics")
        prefix = _exact_token_ids(metrics.get("prompt_token_ids"), "prompt")
        generated = _exact_token_ids(metrics.get("completion_token_ids"), "completion")
        copied = _is_copied_context_step(step)
        probabilities = metrics.get("logprobs")
        if copied and probabilities is None:
            probabilities = [0.0] * len(generated)
        if not isinstance(probabilities, list) or len(probabilities) != len(generated):
            raise ValueError("Agent step requires aligned behavior log probabilities")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value > 0
            for value in probabilities
        ):
            raise ValueError("Behavior log probabilities must be finite and nonpositive")
        if prompt is None:
            prompt = list(prefix)
            sequence = list(prefix)
        if prefix[: len(sequence)] != sequence:
            raise ValueError("Causal prefix was rewritten; this trace cannot be joined for TB")
        observations = prefix[len(sequence) :]
        sequence.extend(observations)
        masks.extend([0] * len(observations))
        logprobs.extend([0.0] * len(observations))
        sequence.extend(generated)
        if copied:
            action_mask = [0] * len(generated)
        else:
            action_mask = metrics.get("loss_mask", [1] * len(generated))
        if (
            not isinstance(action_mask, list)
            or len(action_mask) != len(generated)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1)
                for value in action_mask
            )
        ):
            raise ValueError("Agent loss mask must align with generated tokens")
        masks.extend(int(value) for value in action_mask)
        logprobs.extend(float(value) for value in probabilities)
    if prompt is None:
        return None
    return validate_training_payload(
        {
            "prompt_ids": prompt,
            "response_ids": sequence[len(prompt) :],
            "logprobs": logprobs,
            "loss_mask": masks,
            "stop_reason": "completed",
        }
    )


def build_loss_mask(
    token_ids: list[int],
    reasoning_content: str | None,
    tokenizer: Any | None,
    *,
    message_content: str = "",
) -> list[bool]:
    """Build per-token visible-output mask for a single generation.

    Reconstructs the full text as ``<think>{reasoning}</think>{message}``
    and delegates to ``create_think_mask`` for character-position-based
    masking. This avoids brittle BPE prefix matching.

    This helper is for explicit masking paths, not the default Harbor
    TB/SubTB training behavior. Falls back to all-True when no reasoning or
    tokenizer is available.
    """
    n = len(token_ids)
    if not reasoning_content:
        return [True] * n
    if tokenizer is None:
        raise ValueError("Thinking-token masking requires the model tokenizer")

    from synthstats.core.think_mask import create_think_mask

    full_text = f"<think>{reasoning_content}</think>{message_content}"
    return create_think_mask(full_text, token_ids, tokenizer)


def atif_to_trajectory(
    atif: dict[str, Any],
    reward: Reward,
    *,
    tokenizer: Any | None = None,
    mask_thinking: bool = False,
) -> Trajectory:
    """Convert a full ATIF dict + reward into a SynthStats Trajectory.

    By default, TB/SubTB train on the full generated trace, including
    reasoning tokens. Callers that explicitly want visible-output-only masking
    can pass ``mask_thinking=True``.
    """
    messages = atif_to_messages(atif)
    generations = extract_agent_generations(
        atif,
        tokenizer=tokenizer,
        mask_thinking=mask_thinking,
    )
    return Trajectory(
        messages=messages,
        token_ids=[generation.token_ids for generation in generations],
        token_logprobs=[generation.logprobs for generation in generations],
        loss_mask=[generation.loss_mask for generation in generations],
        reward=reward,
        eos_logprobs=[],
    )


def extract_program(atif: dict[str, Any]) -> str | None:
    """Find the submitted program in an ATIF trajectory (scans in reverse)."""
    for step in reversed(atif.get("steps", [])):
        if step.get("source") != "agent":
            continue
        # check tool_calls for submit_program
        for tc in step.get("tool_calls", []):
            fname = tc.get("function_name", "")
            if "submit" in fname.lower() and "program" in fname.lower():
                args = tc.get("arguments", {})
                code = args.get("code") or args.get("program") or args.get("content")
                if code:
                    return str(code)
        # check message content for <submit_program> tag
        msg = step.get("message", "")
        if isinstance(msg, str) and "<submit_program" in msg:
            import re

            match = re.search(r"<submit_program[^>]*>(.*?)</submit_program>", msg, re.DOTALL)
            if match:
                return match.group(1).strip()
    return None


def read_reward_sidecar(reward_path: str) -> Reward | None:
    """Read a pre-computed reward from a sidecar JSON or reward.txt file."""
    reward_file = Path(reward_path)
    companion = reward_file.with_name("reward_artifact.json")
    agent_dir = (
        reward_file.parent.parent / "agent"
        if reward_file.parent.name == "verifier"
        else reward_file.parent
    )
    if (agent_dir / "synthstats_result.json").is_file() and not companion.is_file():
        raise FileNotFoundError(f"Missing SynthStats verifier reward artifact: {companion}")
    if not reward_file.is_file():
        return None

    if reward_file.name in {"reward.json", "reward.txt"}:
        if companion.is_file():
            return read_reward_sidecar(str(companion))

    try:
        if reward_file.suffix == ".json":
            with reward_file.open() as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"reward JSON must be an object, got {type(data).__name__}")
            if reward_file.name == "reward_artifact.json":
                missing = {"reward", "reward_source", "evidence_kind", "truth_owner"} - data.keys()
                if missing:
                    raise ValueError(
                        f"Incomplete verifier reward artifact: missing {sorted(missing)}"
                    )
                raw_reward = data["reward"]
                if (
                    isinstance(raw_reward, bool)
                    or not isinstance(raw_reward, int | float)
                    or not math.isfinite(float(raw_reward))
                ):
                    raise ValueError(f"Invalid verifier reward in {reward_path}: {raw_reward!r}")
                for field in ("reward_source", "evidence_kind", "truth_owner"):
                    if not isinstance(data[field], str) or not data[field].strip():
                        raise ValueError(
                            f"Invalid verifier {field} in {reward_path}: {data[field]!r}"
                        )
            return reward_artifact_to_reward(data, path=reward_path)

        if reward_file.name == "reward.txt":
            reward_json_path = reward_file.with_name("reward.json")
            if reward_json_path.is_file():
                structured_reward = read_reward_sidecar(str(reward_json_path))
                if structured_reward is not None:
                    return structured_reward

        with reward_file.open() as f:
            total = float(f.read().strip())
        return Reward(
            total=total,
            components={"raw": total},
            info=annotate_reward_info(
                {"path": reward_path},
                reward_source="reward_txt",
                source="reward_txt",
            ),
        )
    except (ValueError, json.JSONDecodeError, OSError) as e:
        if reward_file.name == "reward_artifact.json":
            raise
        logger.warning("failed to read reward sidecar %s: %s", reward_path, e)
        return None
