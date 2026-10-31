"""Typed extensions to the SkyRL 0.4 training configuration."""

from dataclasses import dataclass, field
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

try:
    from skyrl.train.config import (
        AlgorithmConfig,
        GeneratorConfig,
        SkyRLTrainConfig,
        TrainerConfig,
    )

    _SKYRL_CONFIG_AVAILABLE = True
except ImportError:
    _SKYRL_CONFIG_AVAILABLE = False

    @dataclass
    class AlgorithmConfig:  # type: ignore[no-redef]
        policy_loss_type: str = "gfn_tb"
        advantage_estimator: str = "tb_log_reward"
        advantage_batch_normalize: bool = False
        max_seq_len: int = 4096

    @dataclass
    class GeneratorConfig:  # type: ignore[no-redef]
        max_turns: int = 20
        sampling_params: dict[str, Any] = field(default_factory=dict)
        inference_engine: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class TrainerConfig:  # type: ignore[no-redef]
        seed: int = 42
        epochs: int = 1
        train_batch_size: int = 4
        strategy: str = "fsdp"
        ckpt_path: str = "checkpoints"
        policy: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class SkyRLTrainConfig:  # type: ignore[no-redef]
        data: dict[str, Any] = field(default_factory=dict)
        environment: dict[str, Any] = field(default_factory=dict)


@dataclass
class GFNAlgorithmConfig(AlgorithmConfig):
    use_kl_loss: bool = False
    reward_floor: float | None = None
    logZ_init: float = 0.0
    logZ_lr: float = 0.01
    tb_max_residual: float = 100.0
    subtb_lambda: float = 0.9
    lineage_weighting: dict | None = None
    lineage_batching: dict | None = None
    lineage_loss_weight: float = 1.0
    lineage_loss_weights: list[float] | None = None
    microbatch_weight: float | None = None
    dp_size: int = 1
    minibatch_valid_trajectories: int | None = None


@dataclass
class HarborGeneratorConfig(GeneratorConfig):
    type: str = "harbor"
    store_all_messages: bool = True
    enable_summarize: bool = False
    agent_import_path: str = "synthstats.integrations.harbor.agent:SynthStatsInstalledAgent"
    agent_model_name: str | None = None
    sandbox_provider: str = "docker"
    inference_port: int = 8000
    llm_api_base: str | None = None
    agent_timeout_sec: float = 300.0
    collect_rollout_details: bool = True
    task_config_overrides: dict | None = None


@dataclass
class GFNTrainerConfig(TrainerConfig):
    algorithm: GFNAlgorithmConfig = field(default_factory=GFNAlgorithmConfig)
    logger: str = "console"


@dataclass
class TrainingConfig(SkyRLTrainConfig):
    trainer: GFNTrainerConfig = field(default_factory=GFNTrainerConfig)
    generator: HarborGeneratorConfig = field(default_factory=HarborGeneratorConfig)


def build_native_config(config: DictConfig) -> TrainingConfig:
    """Fill native defaults and reject unsupported fields before starting Ray."""
    raw = OmegaConf.to_container(config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("SkyRL configuration must be a mapping")
    raw.pop("type", None)
    raw.pop("model_info", None)
    algorithm = raw["trainer"]["algorithm"]
    for name in ("logZ_init", "logZ_lr"):
        if name in raw:
            algorithm.setdefault(name, raw.pop(name))
    data = raw.get("data", {})
    for name in ("train_data", "val_data"):
        paths = data.get(name, [])
        if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
            raise ValueError(f"runner.data.{name} must be a list of Parquet paths")
    if not data.get("train_data"):
        raise ValueError("runner.data.train_data requires at least one Parquet path")

    policy_cfg = raw.get("trainer", {}).get("policy", {})
    if isinstance(policy_cfg, dict) and isinstance(policy_cfg.get("language_model_only"), bool):
        gen_cfg = raw.setdefault("generator", {})
        if isinstance(gen_cfg, dict):
            ie_cfg = gen_cfg.setdefault("inference_engine", {})
            if isinstance(ie_cfg, dict):
                ie_cfg.setdefault("language_model_only", policy_cfg["language_model_only"])

    gen_raw = raw.get("generator")
    if isinstance(gen_raw, dict):
        overrides = gen_raw.get("task_config_overrides")
        if isinstance(overrides, dict) and isinstance(overrides.get("env"), dict):
            from synthstats.integrations.harbor.task_registry import (
                REGISTERED_TASK_PACKAGES,
                build_local_task_request,
                infer_task_name,
            )

            try:
                task_name = infer_task_name(overrides)
            except ValueError:
                task_name = None
            if task_name in REGISTERED_TASK_PACKAGES:
                local_req = build_local_task_request(overrides)
                merged_overrides = dict(overrides)
                merged_overrides.setdefault("task_name", local_req["task_name"])
                merged_overrides.setdefault("task_path", local_req["task_path"])
                merged_overrides["task_config"] = local_req["task_config"]
                gen_raw["task_config_overrides"] = merged_overrides

    if not _SKYRL_CONFIG_AVAILABLE or not hasattr(TrainingConfig, "from_dict_config"):
        return cast(TrainingConfig, OmegaConf.create(raw))

    try:
        from skyrl.train.utils.utils import validate_cfg
    except ImportError:
        return cast(TrainingConfig, OmegaConf.create(raw))

    native_cfg = cast(TrainingConfig, TrainingConfig.from_dict_config(OmegaConf.create(raw)))
    from synthstats.integrations.skyrl.registry import register_all

    register_all()
    validate_cfg(native_cfg)
    return native_cfg
