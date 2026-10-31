"""SkyRL runner: Hydra CLI entrypoint (runner=skyrl).

Uses SkyRL's ``BasePPOExp`` for proper experiment orchestration:
tokenizer loading, placement groups, inference engine setup,
and trainer instantiation.

Calls ``initialize_ray(cfg)`` (not raw ``ray.init()``) so that
NCCL/vLLM env vars and registry sync are handled by SkyRL.
"""

from __future__ import annotations

import logging
import os
import platform
import sys

from omegaconf import DictConfig

from synthstats.core.import_helpers import try_import_symbol
from synthstats.integrations.skyrl.registry import register_all
from synthstats.integrations.skyrl.trainer import (
    SKYRL_TRAINER_AVAILABLE,
    GFlowNetTrainer,
)
from synthstats.train.experiment_config import (
    build_training_experiment_config,
    to_plain_config,
    validate_training_experiment_config,
)
from synthstats.train.runners.base import RunResult
from synthstats.train.runners.harbor import extract_harbor_generator_kwargs
from synthstats.train.runners.metrics import extract_trainer_metrics

logger = logging.getLogger(__name__)


def _skyrl_install_hint() -> str:
    if platform.system() != "Linux":
        return (
            "This repo only enables synthstats[skyrl] on Linux. "
            "Run this backend on Linux and install it with: uv sync --extra skyrl"
        )
    return "Install with: uv sync --extra skyrl"


# --- SkyRL experiment imports (import-guarded) ---

_BasePPOExp = try_import_symbol(
    "BasePPOExp",
    "skyrl.train.entrypoints.main_base",
)
_initialize_ray = try_import_symbol(
    "initialize_ray",
    "skyrl.train.entrypoints.main_base",
)
if _BasePPOExp is None or _initialize_ray is None:
    logger.debug("SkyRL entrypoints not available; SkyRLRunner will fail at init")


def _make_ppo_exp_class(
    trainer_cls: type,
    generator_cls: type | None = None,
) -> type | None:
    """Build a ``BasePPOExp`` subclass with custom trainer and optional generator.

    Returns ``None`` when ``BasePPOExp`` is not importable (SkyRL missing).
    """
    if _BasePPOExp is None:
        return None

    class SynthStatsPPOExp(_BasePPOExp):  # type: ignore[misc, valid-type]
        """Experiment wrapper that plugs GFlowNetTrainer and optional
        Harbor generator into SkyRL."""

        def get_trainer(self, **kwargs):  # type: ignore[override]
            return trainer_cls(**kwargs)

        def get_generator(self, cfg, tokenizer, inference_engine_client):
            if generator_cls is not None:
                # pass Harbor-specific config from the experiment config
                gen_cfg = cfg.generator
                harbor_kwargs = extract_harbor_generator_kwargs(gen_cfg)
                return generator_cls(
                    cfg,
                    tokenizer=tokenizer,
                    inference_engine_client=inference_engine_client,
                    **harbor_kwargs,
                )
            return super().get_generator(cfg, tokenizer, inference_engine_client)

    return SynthStatsPPOExp


def resolve_harbor_generator_class(exp_cfg: object) -> type | None:
    """Resolve the Harbor generator when the config requests it.

    Lives in the integrations namespace so the import from
    ``synthstats.integrations.harbor.generator`` is architecturally legal
    (``train/*`` must not import ``integrations/*``).
    """
    gen_cfg = getattr(exp_cfg, "generator", None)
    if gen_cfg is None:
        return None

    gen_type = getattr(gen_cfg, "type", None)
    if isinstance(gen_cfg, dict):
        gen_type = gen_cfg.get("type")

    if gen_type != "harbor":
        return None

    from synthstats.integrations.harbor.generator import (
        HARBOR_GENERATOR_AVAILABLE,
        SynthStatsHarborGenerator,
    )

    if not HARBOR_GENERATOR_AVAILABLE:
        raise ImportError(
            "generator.type=harbor but Harbor SDK is not installed. "
            "Cannot fall back to SkyRL default generator: it does not "
            "understand SynthStats tasks and would produce meaningless rewards. "
            "Install the Harbor SDK: uv pip install 'harbor>=0.1.45'"
        )
    return SynthStatsHarborGenerator


class SkyRLRunner:
    """Delegates to SkyRL's Ray-based distributed training via BasePPOExp."""

    def __init__(self, cfg: DictConfig) -> None:
        if not SKYRL_TRAINER_AVAILABLE:
            raise ImportError(f"SkyRL runner requires skyrl-train. {_skyrl_install_hint()}")
        if _BasePPOExp is None:
            raise ImportError(
                "SkyRL runner requires skyrl.train.entrypoints.main_base.BasePPOExp. "
                f"Install or upgrade skyrl-train. {_skyrl_install_hint()}"
            )
        if _initialize_ray is None:
            raise ImportError(
                "SkyRL runner requires skyrl.train.entrypoints.main_base.initialize_ray. "
                f"Install or upgrade skyrl-train. {_skyrl_install_hint()}"
            )
        self.cfg = cfg
        prepared = build_training_experiment_config(cfg)
        validate_training_experiment_config(
            prepared,
            root_cfg=to_plain_config(cfg),
        )
        from synthstats.integrations.skyrl.config import build_native_config

        self.exp_cfg = build_native_config(prepared)

    def _resolve_generator_cls(self) -> type | None:
        return resolve_harbor_generator_class(self.exp_cfg)

    def run(self) -> RunResult:
        try:
            # 1. register custom losses locally before Ray init
            register_all()

            # 2. initialize Ray via SkyRL (handles env vars, NCCL, registry sync)
            # Ray's uv hook changes directories; retain the installed environment.
            os.environ["UV_PROJECT_ENVIRONMENT"] = sys.prefix
            _initialize_ray(self.exp_cfg)

            # 3. resolve optional Harbor generator
            generator_cls = self._resolve_generator_cls()

            # 4. build experiment and run
            PPOExpCls = _make_ppo_exp_class(GFlowNetTrainer, generator_cls)
            if PPOExpCls is None:
                raise ImportError("BasePPOExp unavailable after init check")

            exp = PPOExpCls(self.exp_cfg)
            try:
                exp.run()
            except TypeError as exc:
                # constructor signature mismatch between GFlowNetTrainer
                # and the base RayPPOTrainer at the pinned SkyRL rev
                raise TypeError(
                    f"GFlowNetTrainer constructor mismatch with RayPPOTrainer: {exc}. "
                    "BasePPOExp._setup_trainer() passes (cfg, tracker, tokenizer, "
                    "train_dataset, inference_engine_client, generator, colocate_pg, "
                    "eval_dataset) as keyword args. Verify GFlowNetTrainer.__init__ "
                    "accepts these via *args/**kwargs forwarding."
                ) from exc

            trainer = getattr(exp, "trainer", None)
            metrics = extract_trainer_metrics(trainer)

            return RunResult(metrics=metrics)
        except ImportError:
            raise  # missing dependency = hard failure, don't mask
        except (RuntimeError, TypeError, ValueError, AttributeError) as e:
            logger.exception("SkyRL training failed")
            return RunResult(error=str(e))
