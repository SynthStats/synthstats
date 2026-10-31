"""Shared training configuration dataclasses.

These types are consumed by both train/runners/ and integrations/harbor/.
Placing them here keeps each integration free of cross-dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

from synthstats.core.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_ENTROPY_COEF,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    LOG_REWARD_DEFAULT,
    LOGZ_LR_DEFAULT,
    REWARD_FLOOR_DEFAULT,
    SUBTB_LAMBDA_DEFAULT,
    TB_MAX_RESIDUAL_DEFAULT,
)


@dataclass
class TrainConfig:
    steps: int = 1000
    batch_size: int = DEFAULT_BATCH_SIZE
    temperature: float = DEFAULT_TEMPERATURE
    reward_floor: float = REWARD_FLOOR_DEFAULT
    log_interval: int = 10


@dataclass
class ReplayConfig:
    capacity: int = 1000
    ratio: float = 0.5
    mode: Literal["gfn", "simple"] = "gfn"
    prioritized: bool = True
    alpha: float = 1.0


@dataclass
class OptimConfig:
    policy_lr: float = DEFAULT_LEARNING_RATE
    logZ_lr: float = LOGZ_LR_DEFAULT
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0


@dataclass
class ObjectiveConfig:
    loss_type: str = "tb"
    subtb_lambda: float = SUBTB_LAMBDA_DEFAULT
    tb_max_residual: float = TB_MAX_RESIDUAL_DEFAULT
    lineage_weighting: dict[str, Any] = field(default_factory=dict)
    logZ_init: float = 0.0
    entropy_coef: float = DEFAULT_ENTROPY_COEF
    ab_subtb_alpha: float = 0.1
    use_boundary_critic: bool = False
    boundary_critic_hidden_dim: int = 32
    boundary_critic_loss_coef: float = 1.0
    ref_policy_enabled: bool = False
    ref_policy_weight: float = 1.0
    ref_policy_normalize_by_length: bool = True


@dataclass
class CheckpointConfig:
    save_dir: str = "checkpoints"
    every_steps: int = 100
    keep_last: int = 3
    resume_from: str | None = None


@dataclass
class PolicyConfig:
    model_name: str = "mock"
    dtype: str = "bfloat16"
    max_new_tokens: int | None = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    _target_: str | None = None


@dataclass
class WarmstartConfig:
    enabled: bool = False
    data_path: str | None = None
    strip_thinking: bool = False
    dedupe: bool = True
    compute_rewards: bool = False
    log_reward_default: float = LOG_REWARD_DEFAULT
    max_examples: int | None = None
    log_clamp: tuple[float, float] | None = None
    show_progress: bool = True


@dataclass
class BaseRunnerConfig:
    """Fields shared by all runner configs (Local, Harbor, etc.)."""

    seed: int = 42
    device: str = "auto"
    train: TrainConfig = field(default_factory=TrainConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    warmstart: WarmstartConfig = field(default_factory=WarmstartConfig)


_SECRET_KEYS: frozenset[str] = frozenset(
    {"api_key", "api_token", "secret", "password", "credential"}
)
_REDACTED = "***REDACTED***"


def sanitize_config(
    cfg: dict[str, Any],
    *,
    secret_keys: frozenset[str] = _SECRET_KEYS,
) -> dict[str, Any]:
    """Deep-copy config dict with secret values redacted.

    Any key whose lowercase form contains a substring in *secret_keys*
    has its value replaced with ``"***REDACTED***"``.
    """
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        if any(sk in k.lower() for sk in secret_keys):
            out[k] = _REDACTED
        elif isinstance(v, dict):
            out[k] = sanitize_config(v, secret_keys=secret_keys)
        elif isinstance(v, list):
            out[k] = [
                sanitize_config(item, secret_keys=secret_keys) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            out[k] = v
    return out


def parse_log_clamp(raw: Any) -> tuple[float, float] | None:
    """Parse a log_clamp value from config into (low, high) bounds."""
    if raw is None or isinstance(raw, str):
        return None
    try:
        seq = list(raw)
    except TypeError:
        return None
    if len(seq) == 2 and all(isinstance(x, int | float) for x in seq):
        return (float(seq[0]), float(seq[1]))
    return None


def config_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """Walk nested DictConfig/dict by key path, returning default on miss."""
    current = cfg
    for key in keys:
        if current is None:
            return default
        if hasattr(current, "get"):
            current = current.get(key)
        else:
            return default
    return current if current is not None else default


def parse_base_runner_fields(cfg: Any) -> dict[str, Any]:
    """Parse fields common to all runner configs from a Hydra DictConfig.

    Returns a dict of kwargs suitable for constructing BaseRunnerConfig
    (or any subclass).
    """
    from synthstats.core.constants import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_ENTROPY_COEF,
        DEFAULT_LEARNING_RATE,
        DEFAULT_MAX_NEW_TOKENS,
        DEFAULT_SEED,
        DEFAULT_TEMPERATURE,
        LOG_REWARD_DEFAULT,
        LOGZ_LR_DEFAULT,
        REWARD_FLOOR_DEFAULT,
        SUBTB_LAMBDA_DEFAULT,
        TB_MAX_RESIDUAL_DEFAULT,
    )

    train_raw = config_get(cfg, "runner", "train", default={})
    replay_raw = config_get(cfg, "runner", "replay", default={})
    optim_raw = config_get(cfg, "learner", "optim", default={})
    obj_raw = config_get(cfg, "objective", default={})
    ref_raw = config_get(cfg, "objective", "ref_policy", default={})
    ckpt_raw = config_get(cfg, "checkpoint", default={})
    policy_raw = config_get(cfg, "policy", default={})
    gen_raw = config_get(cfg, "policy", "generation", default={})
    ws_raw = config_get(cfg, "sft_warmstart", default={})

    log_clamp = parse_log_clamp(config_get(ws_raw, "log_clamp"))
    replay_mode = str(config_get(replay_raw, "mode", default="gfn")).lower()
    if replay_mode not in {"gfn", "simple"}:
        replay_mode = "gfn"

    return dict(
        seed=config_get(cfg, "seed", default=DEFAULT_SEED),
        device=config_get(cfg, "device", default="auto"),
        train=TrainConfig(
            steps=config_get(train_raw, "steps", default=1000),
            batch_size=config_get(train_raw, "batch_size", default=DEFAULT_BATCH_SIZE),
            temperature=config_get(train_raw, "temperature", default=DEFAULT_TEMPERATURE),
            reward_floor=config_get(train_raw, "reward_floor", default=REWARD_FLOOR_DEFAULT),
            log_interval=config_get(train_raw, "log_interval", default=10),
        ),
        replay=ReplayConfig(
            capacity=config_get(replay_raw, "capacity", default=0),
            ratio=config_get(replay_raw, "ratio", default=0.5),
            mode=cast(Literal["gfn", "simple"], replay_mode),
            prioritized=config_get(replay_raw, "prioritized", default=True),
            alpha=config_get(replay_raw, "alpha", default=1.0),
        ),
        optim=OptimConfig(
            policy_lr=config_get(optim_raw, "policy_lr", default=DEFAULT_LEARNING_RATE),
            logZ_lr=config_get(optim_raw, "logZ_lr", default=LOGZ_LR_DEFAULT),
            weight_decay=config_get(optim_raw, "weight_decay", default=0.0),
            max_grad_norm=config_get(optim_raw, "max_grad_norm", default=1.0),
        ),
        objective=ObjectiveConfig(
            loss_type=config_get(obj_raw, "loss_type", default="tb"),
            subtb_lambda=config_get(obj_raw, "subtb_lambda", default=SUBTB_LAMBDA_DEFAULT),
            tb_max_residual=config_get(obj_raw, "tb_max_residual", default=TB_MAX_RESIDUAL_DEFAULT),
            lineage_weighting=(
                dict(raw_lineage_weighting)
                if isinstance(
                    (raw_lineage_weighting := config_get(obj_raw, "lineage_weighting", default={})),
                    dict,
                )
                else {}
            ),
            logZ_init=config_get(obj_raw, "logZ_init", default=0.0),
            entropy_coef=config_get(obj_raw, "entropy_coef", default=DEFAULT_ENTROPY_COEF),
            ab_subtb_alpha=config_get(obj_raw, "ab_subtb_alpha", default=0.1),
            use_boundary_critic=config_get(obj_raw, "use_boundary_critic", default=False),
            boundary_critic_hidden_dim=config_get(
                obj_raw, "boundary_critic_hidden_dim", default=32
            ),
            boundary_critic_loss_coef=config_get(obj_raw, "boundary_critic_loss_coef", default=1.0),
            ref_policy_enabled=config_get(ref_raw, "enabled", default=False),
            ref_policy_weight=config_get(ref_raw, "weight", default=1.0),
            ref_policy_normalize_by_length=config_get(ref_raw, "normalize_by_length", default=True),
        ),
        checkpoint=CheckpointConfig(
            save_dir=config_get(ckpt_raw, "save_dir", default="checkpoints"),
            every_steps=config_get(ckpt_raw, "every_steps", default=100),
            keep_last=config_get(ckpt_raw, "keep_last", default=3),
            resume_from=config_get(ckpt_raw, "resume_from"),
        ),
        policy=PolicyConfig(
            model_name=config_get(policy_raw, "model_name", default="mock"),
            dtype=config_get(policy_raw, "dtype", default="bfloat16"),
            max_new_tokens=config_get(gen_raw, "max_new_tokens", default=DEFAULT_MAX_NEW_TOKENS),
            temperature=config_get(gen_raw, "temperature", default=DEFAULT_TEMPERATURE),
            _target_=config_get(policy_raw, "_target_"),
        ),
        warmstart=WarmstartConfig(
            enabled=config_get(ws_raw, "enabled", default=False),
            data_path=config_get(ws_raw, "data_path"),
            strip_thinking=config_get(ws_raw, "strip_thinking", default=False),
            dedupe=config_get(ws_raw, "dedupe", default=True),
            compute_rewards=config_get(ws_raw, "compute_rewards", default=False),
            log_reward_default=config_get(ws_raw, "log_reward_default", default=LOG_REWARD_DEFAULT),
            max_examples=config_get(ws_raw, "max_examples"),
            log_clamp=log_clamp,
            show_progress=config_get(ws_raw, "show_progress", default=True),
        ),
    )
