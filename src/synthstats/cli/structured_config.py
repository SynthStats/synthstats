"""Hydra ConfigStore schemas for the training config surface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore


@dataclass
class RunnerSurfaceConfig:
    type: str = "skyrl"
    logZ_init: Any = None
    logZ_lr: Any = None
    trainer: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    generator: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvSurfaceConfig:
    _target_: str | None = None
    name: str | None = None
    max_steps: int | None = None
    task: dict[str, Any] = field(default_factory=dict)
    codec: dict[str, Any] = field(default_factory=dict)
    executors: dict[str, Any] = field(default_factory=dict)
    judge: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeSurfaceConfig:
    _target_: str | None = None
    scalarization_mode: str | None = None
    beta: float | None = None
    loo_gate_mode: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    judges: list[Any] = field(default_factory=list)
    type: str | None = None
    weight: float | None = None


@dataclass
class PolicySurfaceConfig:
    _target_: str | None = None
    model_name: str | None = None
    device: Any = None
    dtype: str | None = None
    allowed_backends: list[str] | None = None
    generation: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    lora: Any = None
    gradient_checkpointing: bool | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectiveSurfaceConfig:
    reward_floor: float | None = None
    type: str | None = None
    loss_type: str = "tb"
    subtb_lambda: float | None = None
    tb_max_residual: float | None = None
    lineage_weighting: dict[str, Any] = field(default_factory=dict)
    lineage_batching: dict[str, Any] = field(default_factory=dict)
    logZ_init: float | None = None
    logZ_lr: float | None = None
    ab_subtb_alpha: float | None = None
    use_boundary_critic: bool | None = None
    boundary_critic_hidden_dim: int | None = None
    boundary_critic_loss_coef: float | None = None
    ref_policy: dict[str, Any] = field(default_factory=dict)
    entropy_coef: float | None = None
    log_sparse_reward: Any = None
    reward_schedule: dict[str, Any] = field(default_factory=dict)
    exploration: dict[str, Any] = field(default_factory=dict)
    behavior_mix: dict[str, Any] = field(default_factory=dict)
    replay_buffer: dict[str, Any] = field(default_factory=dict)
    eos_token_ids: Any = None
    topk_for_eos: int | None = None


@dataclass
class LearnerSurfaceConfig:
    optim: dict[str, Any] = field(default_factory=dict)
    precision: dict[str, Any] = field(default_factory=dict)
    tinker: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoggingSurfaceConfig:
    backend: str = "console"
    project: str | None = None
    entity: Any = None
    tags: list[Any] = field(default_factory=list)
    name: str | None = None


@dataclass
class CheckpointSurfaceConfig:
    save_dir: str = "checkpoints"
    every_steps: int = 100
    keep_last: int = 3
    resume_from: str | None = None


@dataclass
class TrainSurfaceConfig:
    runner: RunnerSurfaceConfig = field(default_factory=RunnerSurfaceConfig)
    backend: str = "vllm"
    env: EnvSurfaceConfig = field(default_factory=EnvSurfaceConfig)
    judge: JudgeSurfaceConfig = field(default_factory=JudgeSurfaceConfig)
    policy: PolicySurfaceConfig = field(default_factory=PolicySurfaceConfig)
    objective: ObjectiveSurfaceConfig = field(default_factory=ObjectiveSurfaceConfig)
    learner: LearnerSurfaceConfig = field(default_factory=LearnerSurfaceConfig)
    logging: LoggingSurfaceConfig = field(default_factory=LoggingSurfaceConfig)
    checkpoint: CheckpointSurfaceConfig = field(default_factory=CheckpointSurfaceConfig)
    seed: int = 42
    device: str = "auto"
    data_dir: str | None = None
    run: dict[str, Any] = field(default_factory=dict)
    codec: dict[str, Any] = field(default_factory=dict)
    distributed: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    task: dict[str, Any] = field(default_factory=dict)


_REGISTERED = False


def register_train_schemas() -> None:
    """Register ConfigStore schemas for the training config surface."""
    global _REGISTERED

    if _REGISTERED:
        return

    cs = ConfigStore.instance()
    cs.store(name="train_schema", node=TrainSurfaceConfig)

    _REGISTERED = True
