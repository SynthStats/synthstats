"""SkyRL-native generator that runs Harbor Trials for sandboxed rollouts.

Uses the 'launch and wait' pattern: the generator does NOT control each
agent turn. Instead, it starts a Harbor Trial with a Docker sandbox,
the host-side Harbor InstalledAgent drives the SynthStats environment,
and the generator collects the ATIF trajectory when done.

A missing Harbor or SkyRL SDK raises an import error.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synthstats.core.config_utils import config_get as _payload_field
from synthstats.core.import_helpers import try_import_symbol
from synthstats.core.reward_semantics import annotate_reward_info
from synthstats.integrations.harbor.converter import (
    extract_causal_token_data,
    read_reward_sidecar,
    validate_training_payload,
)
from synthstats.integrations.harbor.layout import HarborTrialPaths
from synthstats.integrations.harbor.task_registry import (
    REGISTERED_TASK_PACKAGES,
    _has_meaningful_judge_override,
    infer_task_name,
    resolve_task_package_path,
)

logger = logging.getLogger(__name__)

_LOCAL_REPO_ROOT = Path(__file__).resolve().parents[4]
_LOCAL_REPO_MOUNT_TARGET = "/workspace/synthstats"
_LOCAL_REPO_PYTHONPATH = f"{_LOCAL_REPO_MOUNT_TARGET}/src"
_LOCAL_TASK_IMAGE_REPOSITORY = "synthstats-harbor-e2e"


_GeneratorInterface = try_import_symbol("GeneratorInterface", "skyrl.train.generators.base")

# import-guarded Harbor SDK
_HARBOR_AVAILABLE = False
try:
    import harbor  # noqa: F401

    _HARBOR_AVAILABLE = True
except ImportError:
    pass

_LINEAGE_FLAG_TO_COUNT_KEY: dict[str, str] = {
    "has_trace_id": "n_with_trace_id",
    "has_task_id": "n_with_task_id",
    "has_dataset_row": "n_with_dataset_row",
    "has_shared_generation_record": "n_with_shared_generation_record",
    "has_shared_trial_record": "n_with_shared_trial_record",
    "has_shared_trace_record": "n_with_shared_trace_record",
    "has_verifier_reward": "n_with_verifier_reward",
}


@dataclass(frozen=True)
class NormalizedTaskRequest:
    task_kwargs: dict[str, Any]
    task_config: dict[str, Any]
    task_name: str | None = None


@dataclass(frozen=True)
class ExtractedRewardSignal:
    reward: float
    reward_source: str
    evidence_kind: str
    truth_owner: str


def extract_lineage_artifacts_from_atif(atif: Mapping[str, Any]) -> dict[str, Any]:
    """Extract lineage-bearing metadata from an ATIF record."""
    artifacts: dict[str, Any] = {}
    metadata = atif.get("extra", atif.get("metadata"))
    if not isinstance(metadata, Mapping):
        return artifacts

    for key in (
        "shared_generation_record",
        "shared_trial_record",
        "shared_trace_record",
    ):
        value = metadata.get(key)
        if isinstance(value, Mapping):
            artifacts[key] = dict(value)

    dataset_row = metadata.get("dataset_row")
    if isinstance(dataset_row, Mapping):
        dataset_identity = dict(dataset_row)
        artifacts["dataset_row"] = dataset_identity
        trace_id = dataset_identity.get("trace_id")
        if isinstance(trace_id, str) and trace_id:
            artifacts["trace_id"] = trace_id
        task_id = dataset_identity.get("env_id")
        if isinstance(task_id, str) and task_id:
            artifacts["task_id"] = task_id

    return artifacts


def empty_rollout_lineage_metrics() -> dict[str, int]:
    return {
        "n_with_trace_id": 0,
        "n_with_task_id": 0,
        "n_with_dataset_row": 0,
        "n_with_shared_generation_record": 0,
        "n_with_shared_trial_record": 0,
        "n_with_shared_trace_record": 0,
        "n_with_verifier_reward": 0,
    }


def empty_rollout_lineage_flags() -> dict[str, list[int]]:
    return {key: [] for key in _LINEAGE_FLAG_TO_COUNT_KEY}


def empty_rollout_metrics() -> dict[str, Any]:
    metrics: dict[str, Any] = empty_rollout_lineage_metrics()
    metrics.update(empty_rollout_lineage_flags())
    metrics["n_success"] = 0
    metrics["n_failed"] = 0
    metrics["reward_sources"] = []
    metrics["reward_evidence_kinds"] = []
    metrics["reward_truth_owners"] = []
    return metrics


def extract_rollout_lineage_flags(
    *,
    artifacts: Mapping[str, Any],
    reward_signal: ExtractedRewardSignal,
) -> dict[str, int]:
    return {
        "has_trace_id": int(
            isinstance(artifacts.get("trace_id"), str) and bool(artifacts["trace_id"])
        ),
        "has_task_id": int(
            isinstance(artifacts.get("task_id"), str) and bool(artifacts["task_id"])
        ),
        "has_dataset_row": int(isinstance(artifacts.get("dataset_row"), Mapping)),
        "has_shared_generation_record": int(
            isinstance(artifacts.get("shared_generation_record"), Mapping)
        ),
        "has_shared_trial_record": int(isinstance(artifacts.get("shared_trial_record"), Mapping)),
        "has_shared_trace_record": int(isinstance(artifacts.get("shared_trace_record"), Mapping)),
        "has_verifier_reward": int(reward_signal.truth_owner == "verifier"),
    }


def record_rollout_lineage_metrics(
    metrics: dict[str, Any],
    *,
    artifacts: Mapping[str, Any],
    reward_signal: ExtractedRewardSignal,
) -> dict[str, int]:
    flags = extract_rollout_lineage_flags(
        artifacts=artifacts,
        reward_signal=reward_signal,
    )
    for flag_key, count_key in _LINEAGE_FLAG_TO_COUNT_KEY.items():
        metrics[count_key] += flags[flag_key]
    return flags


def _deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged


def _apply_task_request_overrides(
    *,
    task_kwargs: dict[str, Any],
    task_config: dict[str, Any],
    task_name: str | None,
    overrides: Mapping[str, Any] | None,
    default_seed: int | None,
) -> NormalizedTaskRequest:
    resolved_kwargs = dict(task_kwargs)
    resolved_config = dict(task_config)
    resolved_name = task_name

    seed: int | None = (
        default_seed
        if isinstance(default_seed, int) and not isinstance(default_seed, bool)
        else None
    )
    if isinstance(overrides, Mapping) and overrides:
        raw_seed = overrides.get("seed")
        if isinstance(raw_seed, int) and not isinstance(raw_seed, bool):
            seed = raw_seed

        nested_cfg = overrides.get("task_config")
        override_env = overrides.get("env")
        if not isinstance(override_env, Mapping) and isinstance(nested_cfg, Mapping):
            override_env = nested_cfg.get("env")

        override_judge = overrides.get("judge")
        if not isinstance(override_judge, Mapping) and isinstance(nested_cfg, Mapping):
            override_judge = nested_cfg.get("judge")

        if isinstance(override_env, Mapping) and override_env:
            base_env = resolved_config.get("env")
            merged_env = (
                _deep_merge_dicts(base_env, override_env)
                if isinstance(base_env, Mapping)
                else dict(override_env)
            )
            resolved_config["env"] = merged_env

            override_task_name = overrides.get("task_name")
            if not isinstance(override_task_name, str) or not override_task_name:
                try:
                    override_task_name = infer_task_name({"env": dict(override_env)})
                except ValueError:
                    override_task_name = None
            if isinstance(override_task_name, str) and override_task_name:
                resolved_name = override_task_name
                if override_task_name in REGISTERED_TASK_PACKAGES and "path" in resolved_kwargs:
                    resolved_kwargs["path"] = str(resolve_task_package_path(override_task_name))

        if isinstance(override_judge, Mapping) and _has_meaningful_judge_override(
            dict(override_judge)
        ):
            from synthstats.envs.builders import merge_judge_configs, normalize_judge_config

            base_judge = resolved_config.get("judge")
            resolved_config["judge"] = (
                merge_judge_configs(base_judge, override_judge)
                if isinstance(base_judge, Mapping)
                else normalize_judge_config(override_judge)
            )

    if seed is not None:
        resolved_config["seed"] = seed
        env_section = resolved_config.get("env")
        if isinstance(env_section, Mapping):
            merged_env = dict(env_section)
            if "seed" in merged_env:
                merged_env["seed"] = seed
            if isinstance(merged_env.get("config"), Mapping):
                merged_env_cfg = dict(merged_env["config"])
                merged_env_cfg["evaluation_seed"] = seed
                merged_env["config"] = merged_env_cfg
            resolved_config["env"] = merged_env

    return NormalizedTaskRequest(
        task_kwargs=resolved_kwargs,
        task_config=resolved_config,
        task_name=resolved_name,
    )


def normalize_task_request(
    task_request: dict[str, Any],
    *,
    overrides: Mapping[str, Any] | None = None,
    default_seed: int | None = None,
) -> NormalizedTaskRequest:
    """Normalize SkyRL env_extras into Harbor TaskConfig kwargs + env payload."""
    if not isinstance(task_request, dict):
        raise TypeError(
            "SynthStatsHarborGenerator expects env_extras entries to be dicts, "
            f"got {type(task_request).__name__}"
        )

    task_path = task_request.get("task_path")
    if isinstance(task_path, str) and task_path:
        task_kwargs = {"path": task_path}
        for key in ("git_url", "git_commit_id", "source"):
            value = task_request.get(key)
            if value is not None:
                task_kwargs[key] = value
        task_config = task_request.get("task_config", {})
        if not isinstance(task_config, dict):
            raise TypeError(
                "task_config must be a dict when task_path is provided; "
                f"got {type(task_config).__name__}"
            )
        task_name = task_request.get("task_name")
        return _apply_task_request_overrides(
            task_kwargs=task_kwargs,
            task_config=task_config,
            task_name=task_name if isinstance(task_name, str) else None,
            overrides=overrides,
            default_seed=default_seed,
        )

    return _apply_task_request_overrides(
        task_kwargs=task_request,
        task_config=task_request,
        task_name=task_request.get("name") if isinstance(task_request.get("name"), str) else None,
        overrides=overrides,
        default_seed=default_seed,
    )


def build_task_provenance(task_request: NormalizedTaskRequest) -> dict[str, Any]:
    """Build a stable task-reference payload for Harbor-side reward artifacts."""
    provenance: dict[str, Any] = {}

    if task_request.task_name:
        provenance["task_name"] = task_request.task_name

    path = task_request.task_kwargs.get("path")
    if isinstance(path, str) and path:
        provenance["path"] = path

    for key in ("git_url", "git_commit_id", "source"):
        value = task_request.task_kwargs.get(key)
        if isinstance(value, str) and value:
            provenance[key] = value

    return provenance


def extract_generation_data_from_agent_metadata(
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract token IDs, masks, and logprobs from Harbor agent metadata."""
    token_ids = metadata.get("token_ids")
    mask_ids = metadata.get("mask_ids")
    if not isinstance(token_ids, list) or not isinstance(mask_ids, list):
        return None

    return validate_training_payload(
        {
            "prompt_ids": metadata.get("prompt_ids"),
            "response_ids": token_ids,
            "logprobs": metadata.get("logprobs"),
            "loss_mask": mask_ids,
            "stop_reason": str(metadata.get("stop_reason", "completed")),
        }
    )


def resolve_local_repo_mount_config(
    task_kwargs: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, dict[str, str]]:
    """Mount the local repo or installed package root for local Harbor task packages."""
    task_path = task_kwargs.get("path")
    if not isinstance(task_path, str) or not task_path:
        return None, {}

    resolved_task_path = Path(task_path).resolve()
    base_file = Path(__file__).resolve()
    package_dir = base_file.parents[2]
    import_root = base_file.parents[3]
    repo_root = base_file.parents[4]

    if (repo_root / "src" / "synthstats").is_dir() and import_root == repo_root / "src":
        task_root = repo_root
        mount_source = repo_root
        pythonpath = _LOCAL_REPO_PYTHONPATH
    elif (import_root / "synthstats").is_dir() and package_dir == import_root / "synthstats":
        task_root = package_dir
        mount_source = import_root
        pythonpath = _LOCAL_REPO_MOUNT_TARGET
    else:
        return None, {}

    try:
        resolved_task_path.relative_to(task_root)
    except ValueError:
        return None, {}

    mounts = [
        {
            "type": "bind",
            "source": str(mount_source),
            "target": _LOCAL_REPO_MOUNT_TARGET,
            "read_only": True,
            "bind": {"create_host_path": False},
        }
    ]
    return mounts, {"PYTHONPATH": pythonpath}


def should_delete_trial_environment(
    task_request: NormalizedTaskRequest,
) -> bool:
    """Return whether Harbor should delete task images after the trial.

    Local packaged SynthStats tasks should keep their images so repeated Harbor
    trials can reuse them instead of rebuilding or re-pulling between runs.
    """
    if task_request.task_name in REGISTERED_TASK_PACKAGES:
        return False

    repo_mounts, _ = resolve_local_repo_mount_config(task_request.task_kwargs)
    return repo_mounts is None


def _local_task_image_name(task_name: str) -> str:
    return f"{_LOCAL_TASK_IMAGE_REPOSITORY}:{task_name}"


def ensure_local_task_image(task_request: NormalizedTaskRequest) -> None:
    """Prebuild local Harbor task-package images so Harbor can use prebuilt mode.

    Harbor switches away from its compose-build path when `task.toml` declares
    `environment.docker_image`. For local packaged SynthStats tasks, build that
    stable image up front so repeated trials reuse it.
    """
    task_name = task_request.task_name
    if task_name is None or task_name not in REGISTERED_TASK_PACKAGES:
        return

    task_path = task_request.task_kwargs.get("path")
    if not isinstance(task_path, str) or not task_path:
        return

    environment_dir = Path(task_path).resolve() / "environment"
    if not environment_dir.is_dir():
        raise ValueError(f"Expected Harbor task package environment directory at {environment_dir}")

    image_name = _local_task_image_name(task_name)
    docker_env = os.environ.copy()
    docker_env.setdefault("DOCKER_BUILDKIT", "0")
    docker_env.setdefault("COMPOSE_DOCKER_CLI_BUILD", "0")

    try:
        subprocess.run(
            ["docker", "image", "inspect", image_name],
            check=True,
            env=docker_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    except Exception:
        logger.info("Building missing local Harbor task image %s", image_name)

    subprocess.run(
        ["docker", "build", "-t", image_name, str(environment_dir)],
        check=True,
        env=docker_env,
    )


def _harbor_sdk_install_hint() -> str:
    return "Install Harbor with: uv sync --extra harbor"


def _harbor_generator_install_hint() -> str:
    return (
        "The Harbor training generator requires Linux and "
        "uv sync --extra skyrl --extra ml --extra pymc"
    )


def resolve_llm_api_base(
    llm_api_base: str | None,
    *,
    inference_port: int,
    platform_name: str | None = None,
) -> str:
    """Resolve the Harbor agent's LLM endpoint.

    Harbor invokes this InstalledAgent on the host process, so the derived default
    must be host-reachable rather than container-reachable.
    """
    if llm_api_base:
        return llm_api_base

    _ = platform_name or platform.system()
    return f"http://127.0.0.1:{inference_port}/v1"


_SUPPORTED_SAMPLING_PARAMS: frozenset[str] = frozenset(
    {
        "max_generate_length",
        "max_tokens",
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "stop",
        "stop_token_ids",
        "include_stop_str_in_output",
        "min_tokens",
        "skip_special_tokens",
        "repetition_penalty",
        "logprobs",
        "additional_kwargs",
    }
)


def _normalize_sampling_source(source: Any) -> dict[str, Any]:
    """Validate and normalize a sampling-parameter container."""
    if source is None:
        return {}
    if isinstance(source, Mapping):
        raw = {str(key): value for key, value in source.items()}
        is_mapping = True
    else:
        raw = {key: value for key, value in vars(source).items() if not key.startswith("_")}
        is_mapping = False

    unknown = sorted(set(raw) - _SUPPORTED_SAMPLING_PARAMS)
    if unknown:
        raise ValueError(f"Unsupported sampling parameter(s): {', '.join(unknown)}")

    additional_kwargs = raw.pop("additional_kwargs", None)
    if not is_mapping and raw.get("repetition_penalty") == 1.0:
        raw.pop("repetition_penalty", None)

    if additional_kwargs is not None:
        if not isinstance(additional_kwargs, Mapping):
            raise ValueError("Unsupported sampling parameter: additional_kwargs must be a mapping")
        extra = {str(key): value for key, value in additional_kwargs.items()}
        unknown_extra = sorted(set(extra) - (_SUPPORTED_SAMPLING_PARAMS - {"additional_kwargs"}))
        if unknown_extra:
            raise ValueError(f"Unsupported sampling parameter(s): {', '.join(unknown_extra)}")
        for key, value in extra.items():
            raw[key] = value

    if "logprobs" in raw and raw["logprobs"] not in (None, 1):
        raise ValueError(
            f"Unsupported sampling parameter logprobs={raw['logprobs']!r}; "
            "token rollout requires logprobs=1"
        )

    if raw.get("stop") is not None:
        stop_val = raw["stop"]
        if isinstance(stop_val, str):
            raw["stop"] = [stop_val]
        elif isinstance(stop_val, Sequence) and not isinstance(stop_val, (bytes, bytearray)):
            if any(not isinstance(item, str) for item in stop_val):
                raise ValueError("Unsupported sampling parameter: stop must contain only strings")
            raw["stop"] = list(stop_val)
        else:
            raise ValueError(
                "Unsupported sampling parameter: stop must be a string or sequence of strings"
            )

    if raw.get("stop_token_ids") is not None:
        stop_token_ids = raw["stop_token_ids"]
        if not isinstance(stop_token_ids, Sequence) or isinstance(
            stop_token_ids, (str, bytes, bytearray)
        ):
            raise ValueError(
                "Unsupported sampling parameter: stop_token_ids must be a sequence of integers"
            )
        if any(isinstance(tok, bool) or not isinstance(tok, int) for tok in stop_token_ids):
            raise ValueError(
                "Unsupported sampling parameter: stop_token_ids must contain only integers"
            )
        raw["stop_token_ids"] = [int(tok) for tok in stop_token_ids]

    if raw.get("include_stop_str_in_output") is not None and not isinstance(
        raw["include_stop_str_in_output"], bool
    ):
        raise ValueError(
            "Unsupported sampling parameter: include_stop_str_in_output must be a bool"
        )

    return raw


def _resolve_max_generate_length(source: Mapping[str, Any]) -> Any:
    for key in ("max_generate_length", "max_tokens", "max_new_tokens"):
        val = source.get(key)
        if val is not None:
            return val
    return None


def resolve_trial_sampling_params(
    default_sampling: Any,
    batch_sampling: Any = None,
) -> dict[str, Any]:
    """Resolve per-trial sampling parameters from batch input with config fallback."""
    default_cfg = _normalize_sampling_source(default_sampling)
    default_max_gen = _resolve_max_generate_length(default_cfg)

    has_batch_sampling = batch_sampling is not None and (
        not isinstance(batch_sampling, Mapping) or bool(batch_sampling)
    )
    batch_cfg = _normalize_sampling_source(batch_sampling) if has_batch_sampling else {}

    max_gen = _resolve_max_generate_length(batch_cfg) if has_batch_sampling else None
    if max_gen is None:
        max_gen = default_max_gen

    def _pick(field: str) -> Any:
        if has_batch_sampling:
            val = batch_cfg.get(field)
            if val is not None:
                return val
        return default_cfg.get(field)

    resolved: dict[str, Any] = {
        "max_generate_length": max_gen,
        "temperature": _pick("temperature"),
        "top_p": _pick("top_p"),
        "top_k": _pick("top_k"),
    }
    for optional_key in (
        "min_p",
        "stop",
        "stop_token_ids",
        "include_stop_str_in_output",
        "min_tokens",
        "skip_special_tokens",
        "repetition_penalty",
    ):
        val = _pick(optional_key)
        if val is not None:
            resolved[optional_key] = val
    return resolved


def extract_generation_data_from_atif(atif: dict[str, Any]) -> dict[str, Any] | None:
    """Extract token IDs, logprobs, and loss masks from ATIF.

    Returns data formatted for GeneratorOutput: all lists use ints
    for loss_mask (0/1, not bool) per SkyRL's TypedDict spec.
    """
    return extract_causal_token_data(atif)


def _reward_signal(
    reward: float,
    *,
    reward_info: dict[str, Any] | None,
    default_source: str,
) -> ExtractedRewardSignal:
    info = annotate_reward_info(
        reward_info,
        source=default_source,
    )
    return ExtractedRewardSignal(
        reward=float(reward),
        reward_source=str(info["reward_source"]),
        evidence_kind=str(info["evidence_kind"]),
        truth_owner=str(info["truth_owner"]),
    )


def extract_reward_signal_from_trial_result(trial_result: Any) -> ExtractedRewardSignal:
    """Extract a scalar reward and canonical semantics from Harbor trial artifacts."""

    trial_uri = _payload_field(trial_result, "trial_uri")
    if trial_uri is not None:
        task_dir = HarborTrialPaths.from_uri(trial_uri).verifier
        for name in ("reward_artifact.json", "reward.json"):
            structured_reward = read_reward_sidecar(str(task_dir / name))
            if structured_reward is not None:
                return _reward_signal(
                    structured_reward.total,
                    reward_info=structured_reward.info,
                    default_source=name,
                )

    verifier = _payload_field(trial_result, "verifier_result")
    if verifier is not None:
        rewards = _payload_field(verifier, "rewards")
        if rewards is not None:
            reward = _payload_field(rewards, "reward")
            if reward is not None:
                return _reward_signal(
                    reward,
                    reward_info=None,
                    default_source="verifier_result",
                )

    if trial_uri is not None:
        task_dir = HarborTrialPaths.from_uri(trial_uri).verifier
        for name in ("reward.txt",):
            sidecar_reward = read_reward_sidecar(str(task_dir / name))
            if sidecar_reward is not None:
                return _reward_signal(
                    sidecar_reward.total,
                    reward_info=sidecar_reward.info,
                    default_source=name,
                )

    agent_result = _payload_field(trial_result, "agent_result")
    if agent_result is not None:
        metadata = _payload_field(agent_result, "metadata")
        reward = _payload_field(agent_result, "reward")
        reward_info = _payload_field(agent_result, "reward_info")
        if reward is not None:
            info = reward_info if isinstance(reward_info, dict) else None
            if info is None and isinstance(metadata, dict):
                metadata_info = metadata.get("reward_info")
                metadata_reward = metadata.get("reward")
                if isinstance(metadata_info, dict) and (
                    metadata_reward is None or metadata_reward == reward
                ):
                    info = metadata_info
            return _reward_signal(
                reward,
                reward_info=info,
                default_source="agent_result",
            )

        if isinstance(metadata, dict) and "reward" in metadata:
            info = metadata.get("reward_info")
            return _reward_signal(
                metadata["reward"],
                reward_info=info if isinstance(info, dict) else None,
                default_source="agent_metadata",
            )

        metrics = _payload_field(agent_result, "metrics", {})
        if isinstance(metrics, dict) and "reward" in metrics:
            info = None
            if isinstance(metadata, dict):
                metadata_info = metadata.get("reward_info")
                metadata_reward = metadata.get("reward")
                if isinstance(metadata_info, dict) and (
                    metadata_reward is None or metadata_reward == metrics["reward"]
                ):
                    info = metadata_info
            return _reward_signal(
                metrics["reward"],
                reward_info=info,
                default_source="agent_metrics",
            )

    logger.warning(
        "Could not extract reward from trial_result: %r; defaulting to 0.0",
        trial_result,
    )
    return _reward_signal(
        0.0,
        reward_info=None,
        default_source="default",
    )


def extract_reward_from_trial_result(trial_result: Any) -> tuple[float, str]:
    """Extract a scalar reward and its provenance from Harbor trial artifacts."""
    signal = extract_reward_signal_from_trial_result(trial_result)
    return signal.reward, signal.reward_source


def _make_generator_class():
    """Build SynthStatsHarborGenerator dynamically to avoid import errors."""
    if _GeneratorInterface is None:
        return None

    class SynthStatsHarborGenerator(_GeneratorInterface):  # type: ignore[misc, valid-type]
        """Runs Harbor Trials as SkyRL generator rollouts.

        Each call to generate() creates one Harbor Trial per task in the batch,
        runs them concurrently, and collects ATIF trajectories + rewards.
        The agent inside the Docker container calls SkyRL's HTTP endpoint
        for LLM inference.
        """

        def __init__(
            self,
            config: Any,
            *args: Any,
            agent_import_path: str = (
                "synthstats.integrations.harbor.agent:SynthStatsInstalledAgent"
            ),
            agent_model_name: str | None = None,
            sandbox_provider: str = "docker",
            inference_port: int = 8000,
            llm_api_base: str | None = None,
            agent_timeout_sec: float = 300.0,
            collect_rollout_details: bool = True,
            **kwargs: Any,
        ) -> None:
            if not _HARBOR_AVAILABLE:
                raise ImportError(
                    "Harbor SDK required for SynthStatsHarborGenerator. "
                    f"{_harbor_sdk_install_hint()}"
                )
            self.cfg = config
            self.inference_engine_client = kwargs.pop("inference_engine_client", None)
            if llm_api_base is None:
                if self.inference_engine_client is None:
                    raise ValueError(
                        "Harbor generation requires an inference client or LLM API URL"
                    )
                endpoint = self.inference_engine_client.get_endpoint_url().rstrip("/")
                llm_api_base = endpoint if endpoint.endswith("/v1") else endpoint + "/v1"
            self.agent_import_path = agent_import_path
            self.agent_model_name = agent_model_name
            self.sandbox_provider = sandbox_provider
            self.inference_port = inference_port
            self.llm_api_base = resolve_llm_api_base(
                llm_api_base,
                inference_port=inference_port,
            )
            self.agent_timeout_sec = agent_timeout_sec
            self.collect_rollout_details = collect_rollout_details
            self._tokenizer = kwargs.pop("tokenizer", None)
            logger.info(
                "SynthStatsHarborGenerator: agent=%s, sandbox=%s, port=%d, llm_api_base=%s",
                agent_import_path,
                sandbox_provider,
                inference_port,
                self.llm_api_base,
            )

        async def generate(self, input_batch: Any) -> dict[str, Any]:
            """Run Harbor Trials for a batch of tasks and return GeneratorOutput.

            Args:
                input_batch: GeneratorInput with env_extras containing task configs

            Returns:
                GeneratorOutput TypedDict with prompt_token_ids, response_ids,
                rewards, loss_masks, stop_reasons, rollout_metrics, rollout_logprobs
            """
            import asyncio

            tasks = self._extract_tasks(input_batch)
            if not tasks:
                return self._empty_output()

            batch_sampling_params = self._extract_sampling_params(input_batch)
            trial_coros = [
                (
                    self._run_single_trial(task, sampling_params=batch_sampling_params)
                    if batch_sampling_params is not None
                    else self._run_single_trial(task)
                )
                for task in tasks
            ]
            results = await asyncio.gather(*trial_coros, return_exceptions=True)

            prompt_token_ids: list[list[int]] = []
            response_ids: list[list[int]] = []
            rewards: list[float] = []
            loss_masks: list[list[int]] = []
            stop_reasons: list[str] = []
            rollout_logprobs: list[list[float]] = []
            rollout_metrics = empty_rollout_metrics()
            reward_sources: list[str] = []
            reward_evidence_kinds: list[str] = []
            reward_truth_owners: list[str] = []

            n_success = 0
            for i, result in enumerate(results):
                if isinstance(result, BaseException):
                    raise RuntimeError(f"Harbor trajectory {i} failed") from result

                trial_result, atif = result
                gen_data = self._extract_generation_data(trial_result, atif)
                if gen_data is None:
                    raise ValueError(f"Harbor trajectory {i} has no token-level training payload")

                prompt_token_ids.append(gen_data["prompt_ids"])
                response_ids.append(gen_data["response_ids"])
                loss_masks.append(gen_data["loss_mask"])
                rollout_logprobs.append(gen_data["logprobs"])

                reward_signal = extract_reward_signal_from_trial_result(trial_result)
                rewards.append(reward_signal.reward)
                reward_sources.append(reward_signal.reward_source)
                reward_evidence_kinds.append(reward_signal.evidence_kind)
                reward_truth_owners.append(reward_signal.truth_owner)
                rollout_flags = record_rollout_lineage_metrics(
                    rollout_metrics,
                    artifacts=extract_lineage_artifacts_from_atif(atif),
                    reward_signal=reward_signal,
                )
                for key, value in rollout_flags.items():
                    rollout_metrics[key].append(value)
                stop_reasons.append(gen_data.get("stop_reason", "completed"))
                n_success += 1

            rollout_metrics["n_success"] = n_success
            rollout_metrics["n_failed"] = len(tasks) - n_success
            rollout_metrics["reward_sources"] = reward_sources
            rollout_metrics["reward_evidence_kinds"] = reward_evidence_kinds
            rollout_metrics["reward_truth_owners"] = reward_truth_owners

            return {
                "prompt_token_ids": prompt_token_ids,
                "response_ids": response_ids,
                "rewards": rewards,
                "loss_masks": loss_masks,
                "stop_reasons": stop_reasons,
                "rollout_metrics": rollout_metrics,
                "rollout_logprobs": rollout_logprobs,
            }

        def _extract_tasks(self, input_batch: Any) -> list[dict[str, Any]]:
            """Extract task configurations from GeneratorInput."""
            env_extras = getattr(input_batch, "env_extras", None)
            if env_extras is None:
                env_extras = (
                    input_batch.get("env_extras", []) if isinstance(input_batch, dict) else []
                )
            if not isinstance(env_extras, list):
                env_extras = [env_extras]
            return env_extras

        def _extract_sampling_params(self, input_batch: Any) -> Any:
            """Extract optional per-call sampling parameters from GeneratorInput."""
            sampling_params = getattr(input_batch, "sampling_params", None)
            if sampling_params is None and isinstance(input_batch, Mapping):
                sampling_params = input_batch.get("sampling_params")
            return sampling_params

        async def _run_single_trial(
            self,
            task_config: dict[str, Any],
            *,
            sampling_params: Any = None,
        ) -> tuple[Any, dict[str, Any]]:
            """Run a single Harbor Trial and return (TrialResult, atif_dict).

            Uses Harbor's TrialConfig with:
            - AgentConfig: import_path, model_name, kwargs, env
            - EnvironmentConfig: provider for sandbox type
            - TaskConfig: the task definition
            """
            from harbor.models.trial.config import (
                AgentConfig,
                EnvironmentConfig,
                TaskConfig,
                TrialConfig,
            )
            from harbor.trial.trial import Trial

            gen_cfg = getattr(self.cfg, "generator", None)
            trainer_cfg = getattr(self.cfg, "trainer", None)
            overrides = _payload_field(gen_cfg, "task_config_overrides")
            default_seed = _payload_field(trainer_cfg, "seed")
            normalized = normalize_task_request(
                task_config,
                overrides=overrides if isinstance(overrides, Mapping) else None,
                default_seed=default_seed if isinstance(default_seed, int) else None,
            )
            ensure_local_task_image(normalized)
            resolved_sampling = resolve_trial_sampling_params(
                self.cfg.generator.sampling_params,
                sampling_params,
            )
            repo_mounts, environment_env = resolve_local_repo_mount_config(normalized.task_kwargs)
            agent_env = {
                "LLM_API_BASE": self.llm_api_base,
                "TASK_CONFIG": json.dumps(normalized.task_config),
                "MODEL_NAME": self.agent_model_name or self.cfg.trainer.policy.model.path,
                "MAX_TURNS": str(self.cfg.generator.max_turns),
                "MAX_SEQUENCE_LENGTH": str(self.cfg.trainer.algorithm.max_seq_len),
                "SAMPLING_PARAMS": json.dumps(resolved_sampling),
                "TASK_PROVENANCE": json.dumps(
                    build_task_provenance(normalized),
                    sort_keys=True,
                ),
            }
            if "PYTHONPATH" in environment_env:
                agent_env["PYTHONPATH"] = environment_env["PYTHONPATH"]
            agent_config = AgentConfig(
                import_path=self.agent_import_path,
                model_name=self.agent_model_name,
                override_timeout_sec=self.agent_timeout_sec,
                kwargs={
                    "collect_rollout_details": self.collect_rollout_details,
                },
                env=agent_env,
            )
            environment_config = EnvironmentConfig(
                type=self.sandbox_provider,
                delete=should_delete_trial_environment(normalized),
                mounts=repo_mounts,
                env=environment_env,
            )
            task_cfg = TaskConfig(**normalized.task_kwargs)
            trial_config = TrialConfig(
                agent=agent_config,
                environment=environment_config,
                task=task_cfg,
            )

            trial = await Trial.create(trial_config)
            result = await trial.run()

            atif = self._extract_atif(result)
            from harbor.models.trajectories.trajectory import Trajectory

            Trajectory.model_validate(atif)
            return result, atif

        def _extract_atif(self, trial_result: Any) -> dict[str, Any]:
            """Extract ATIF trajectory dict from TrialResult.

            TrialResult stores results at trial_uri. The ATIF trajectory
            is accessed via agent_result or by reading from the trial_uri path.
            """
            # try agent_result first (AgentContext may contain trajectory)
            agent_result = _payload_field(trial_result, "agent_result")
            if agent_result is not None:
                trajectory = _payload_field(agent_result, "trajectory")
                if trajectory is not None and isinstance(trajectory, dict):
                    return trajectory

            # read from trial_uri (filesystem path to trial artifacts)
            trial_uri = _payload_field(trial_result, "trial_uri")
            if trial_uri is not None:
                traj_path = HarborTrialPaths.from_uri(trial_uri).trajectory
                if traj_path.exists():
                    with open(traj_path) as f:
                        return json.load(f)

            raise ValueError("Harbor trial has no ATIF trajectory")

        def _extract_generation_data(
            self,
            trial_result: Any,
            atif: dict[str, Any],
        ) -> dict[str, Any] | None:
            agent_result = _payload_field(trial_result, "agent_result")
            metadata = _payload_field(agent_result, "metadata")
            if isinstance(metadata, dict):
                metadata_data = extract_generation_data_from_agent_metadata(metadata)
                if metadata_data is not None:
                    return metadata_data
            return extract_generation_data_from_atif(atif)

        def _empty_output(self) -> dict[str, Any]:
            return {
                "prompt_token_ids": [],
                "response_ids": [],
                "rewards": [],
                "loss_masks": [],
                "stop_reasons": [],
                "rollout_metrics": empty_rollout_metrics(),
                "rollout_logprobs": [],
            }

    return SynthStatsHarborGenerator


_GeneratorClass = _make_generator_class()

HARBOR_GENERATOR_AVAILABLE = _GeneratorClass is not None


def _unavailable_generator(*args: Any, **kwargs: Any) -> None:
    raise ImportError(_harbor_generator_install_hint())


SynthStatsHarborGenerator = (
    _GeneratorClass if _GeneratorClass is not None else _unavailable_generator
)

__all__ = [
    "SynthStatsHarborGenerator",
    "HARBOR_GENERATOR_AVAILABLE",
    "build_task_provenance",
    "empty_rollout_lineage_flags",
    "empty_rollout_metrics",
    "empty_rollout_lineage_metrics",
    "extract_generation_data_from_agent_metadata",
    "extract_generation_data_from_atif",
    "extract_lineage_artifacts_from_atif",
    "extract_rollout_lineage_flags",
    "extract_reward_from_trial_result",
    "normalize_task_request",
    "record_rollout_lineage_metrics",
    "resolve_llm_api_base",
]
