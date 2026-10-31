#!/usr/bin/env python
"""SynthStats training CLI.

Entrypoint for Harbor-native SkyRL training.

Usage:
    # Default Harbor-native SkyRL training
    synthstats-train env=dugongs policy=hf_qwen3_5_0_8b

    # Or via module:
    uv run python -m synthstats.cli.train env=dugongs
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import hydra
from omegaconf import DictConfig, OmegaConf

from synthstats.cli.structured_config import register_train_schemas
from synthstats.train.experiment_config import validate_training_controls

if TYPE_CHECKING:
    from synthstats.train.runners.base import Runner

logger = logging.getLogger(__name__)

_VALID_BACKENDS = {"vllm"}
_VALID_LOSS_TYPES = {"subtb", "tb"}
_VALID_RUNNER_TYPES = {"skyrl", "skyrl_harbor"}


def _bootstrap_environment() -> None:
    """Load local `.env` before Hydra resolves `${oc.env:...}` interpolations."""
    from dotenv import find_dotenv, load_dotenv

    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)


def _policy_allowed_backends(policy_cfg: DictConfig | dict) -> set[str] | None:
    allowed = policy_cfg.get("allowed_backends")
    if allowed is None:
        return None
    if isinstance(allowed, str):
        return {allowed}
    try:
        return {str(item) for item in allowed}
    except TypeError as exc:
        raise ValueError("policy.allowed_backends must be a string or list of strings") from exc


def validate_train_config(cfg: DictConfig) -> None:
    backend = str(cfg.get("backend", "vllm"))
    if backend not in _VALID_BACKENDS:
        valid_options = ", ".join(sorted(_VALID_BACKENDS))
        raise ValueError(
            f"backend={backend} is not a valid backend. Valid options: {valid_options}"
        )

    objective_cfg = cfg.get("objective", {})
    if objective_cfg.get("type") == "tb_subtb" or objective_cfg.get("ab_subtb_alpha") is not None:
        raise ValueError(
            "The TB/SubTB blend (objective=tb_subtb) is not implemented by this runner. "
            "SubTB requires aligned, differentiable current-policy EOS probabilities; "
            "select objective=tb."
        )
    loss_type = str(objective_cfg.get("loss_type", "tb"))
    if loss_type not in _VALID_LOSS_TYPES:
        valid_options = ", ".join(sorted(_VALID_LOSS_TYPES))
        raise ValueError(
            "objective.loss_type="
            f"{loss_type} is not supported by the training CLI. Valid options: {valid_options}"
        )

    if loss_type == "subtb":
        raise ValueError(
            "SubTB requires aligned, differentiable current-policy EOS probabilities "
            "from the policy worker. This runner does not supply them; select objective=tb."
        )

    validate_training_controls(cfg)

    policy_cfg = cfg.get("policy", {})
    allowed_backends = _policy_allowed_backends(policy_cfg)
    if allowed_backends is not None and backend not in allowed_backends:
        valid_options = ", ".join(sorted(allowed_backends))
        model_name = policy_cfg.get("model_name", "<unknown>")
        raise ValueError(
            f"policy.model_name={model_name} does not support backend={backend}. "
            f"Allowed backends: {valid_options}"
        )

    from synthstats.envs.builders import merge_judge_configs

    env_cfg = cfg.get("env", {})
    if env_cfg.get("task", {}).get("language") == "lazyppl" or "lazyppl" in env_cfg.get(
        "executors", {}
    ):
        raise ValueError(
            "LazyPPL training requires GHC, Stack and a configured LazyPPL library checkout "
            "inside the Harbor task container. The bundled task images do not provide them; "
            "select a PyMC environment such as env=dugongs."
        )
    judge_cfg = merge_judge_configs(env_cfg.get("judge"), cfg.get("judge"))
    if _uses_llm_critic(judge_cfg):
        raise ValueError(
            "LLM critic judges are unavailable in the bundled Harbor task images: "
            "LiteLLM and judge credentials are not supplied to the container. "
            "Select judge=likelihood or judge=composite."
        )


def _uses_llm_critic(config: object) -> bool:
    if isinstance(config, Mapping):
        if config.get("_target_") == "synthstats.judges.llm_critic.LLMCriticJudge":
            return True
        if config.get("type") == "llm_critic":
            return True
        return any(_uses_llm_critic(value) for value in config.values())
    if isinstance(config, Sequence) and not isinstance(config, str):
        return any(_uses_llm_critic(value) for value in config)
    return False


def build_runner(cfg: DictConfig) -> Runner:
    runner_cfg = cfg.get("runner", {})
    runner_type = runner_cfg.get("type", "skyrl")
    if runner_type == "skyrl_harbor":
        runner_type = "skyrl"

    if runner_type == "skyrl":
        validate_train_config(cfg)
        from synthstats.integrations.skyrl.runner import SkyRLRunner

        return SkyRLRunner(cfg)

    valid_options = ", ".join(sorted(_VALID_RUNNER_TYPES))
    raise ValueError(
        f"runner={runner_type} is not a valid runner type. Valid options: {valid_options}"
    )


def render_debug_config(cfg: DictConfig) -> str:
    """Render a sanitized debug view of the Hydra config."""
    from synthstats.core.config import sanitize_config

    container = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(container, dict):
        return OmegaConf.to_yaml(cfg)
    sanitized = sanitize_config(container)
    return OmegaConf.to_yaml(OmegaConf.create(sanitized))


register_train_schemas()


@hydra.main(version_base=None, config_path="pkg://synthstats.configs", config_name="train")
def _hydra_main(cfg: DictConfig) -> float | None:
    """Training entrypoint; return final loss for HPO and raise on failure."""
    logger.debug("Config:\n%s", render_debug_config(cfg))

    try:
        runner = build_runner(cfg)
        result = runner.run()

        if result.error:
            raise RuntimeError(f"Training failed: {result.error}")

        if result.interrupted:
            logger.info("Training interrupted")

        if result.checkpoints:
            logger.info("Checkpoints: %s", result.checkpoints)

        logger.info("Final metrics: %s", result.metrics)

        return result.metrics.get("loss")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        raise

    except Exception as e:
        logger.exception("Training failed with exception: %s", e)
        raise


def main() -> float | None:
    """Console entrypoint that bootstraps the environment before Hydra composes config."""
    _bootstrap_environment()
    return _hydra_main()


if __name__ == "__main__":
    main()
