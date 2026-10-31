"""Experiment-config normalization helpers."""

from __future__ import annotations

from copy import deepcopy

from omegaconf import DictConfig, OmegaConf

from synthstats.train.config_validation import validate_gfn_algorithm_config


def to_plain_config(cfg: DictConfig | dict | None) -> dict:
    """Convert OmegaConf-backed configs into plain mutable dicts."""
    if cfg is None:
        return {}
    if OmegaConf.is_config(cfg):
        container = OmegaConf.to_container(cfg, resolve=True)
        return container if isinstance(container, dict) else {}
    if isinstance(cfg, dict):
        return deepcopy(cfg)
    return {}


def infer_policy_loss_type(root_cfg: dict, runner_cfg: dict) -> str | None:
    """Infer the backend loss name from the app-level objective config."""
    del runner_cfg
    objective_cfg = root_cfg.get("objective", {})
    if not isinstance(objective_cfg, dict):
        return None
    loss_type = objective_cfg.get("loss_type")
    if loss_type == "subtb":
        return "gfn_subtb"
    if loss_type == "tb":
        return "gfn_tb"
    return None


def promote_objective_algorithm_settings(
    objective_cfg: dict[str, object],
    algorithm_cfg: dict[str, object],
) -> None:
    """Copy objective-owned trainer settings into the algorithm config."""
    for key in (
        "subtb_lambda",
        "tb_max_residual",
        "reward_floor",
        "lineage_weighting",
        "lineage_batching",
    ):
        value = objective_cfg.get(key)
        if value is not None and key not in algorithm_cfg:
            algorithm_cfg[key] = deepcopy(value)


def cfg_get(container: object, *keys: str, default: object = None) -> object:
    """Walk a nested dict/object config and return a value if present."""
    current = container
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key)
            continue
        if hasattr(current, key):
            current = getattr(current, key)
            continue
        if hasattr(current, "get"):
            current = current.get(key)
            continue
        return default
    return current if current is not None else default


def as_positive_int(value: object) -> int | None:
    """Parse a strictly positive integer while rejecting booleans."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    return None


def validate_training_controls(cfg: DictConfig | dict) -> None:
    """Reject config controls that the training runner cannot execute."""
    entropy_coef = cfg.get("objective", {}).get("entropy_coef")
    if isinstance(entropy_coef, bool) or entropy_coef not in (None, 0.0):
        raise ValueError(
            "objective.entropy_coef is not supported by the TB training runner. "
            "Use objective.entropy_coef=0.0."
        )
    if cfg.get("policy", {}).get("_target_") is not None:
        raise ValueError(
            "policy._target_ is not supported by the training runner, which selects "
            "its local model through policy.model_name."
        )


def validate_training_experiment_config(
    exp_cfg: DictConfig | dict,
    *,
    root_cfg: dict | None = None,
) -> None:
    """Fail fast on invalid SynthStats distributed-training combinations."""
    validate_training_controls(exp_cfg)
    if root_cfg is not None:
        validate_training_controls(root_cfg)
    validate_gfn_algorithm_config(exp_cfg)

    policy_lm_only = cfg_get(exp_cfg, "trainer", "policy", "language_model_only")
    ie_lm_only = cfg_get(exp_cfg, "generator", "inference_engine", "language_model_only")
    if policy_lm_only is True and ie_lm_only is False:
        raise ValueError(
            "language_model_only should be set consistently between inference engine and policy"
        )

    generator_type = cfg_get(exp_cfg, "generator", "type")
    if generator_type != "harbor":
        return

    max_seq_len = cfg_get(exp_cfg, "trainer", "algorithm", "max_seq_len")
    max_seq_len_int = as_positive_int(max_seq_len)
    if max_seq_len_int is None:
        raise ValueError(
            "runner.trainer.algorithm.max_seq_len must be set to a positive integer "
            "for Harbor-backed multi-turn SkyRL training."
        )

    policy_max_new_tokens = None
    if root_cfg is not None:
        policy_max_new_tokens = cfg_get(
            root_cfg,
            "policy",
            "generation",
            "max_new_tokens",
        )
    policy_max_new_tokens_int = as_positive_int(policy_max_new_tokens)
    if policy_max_new_tokens_int is not None and max_seq_len_int < policy_max_new_tokens_int:
        raise ValueError(
            "runner.trainer.algorithm.max_seq_len must be >= "
            "policy.generation.max_new_tokens for Harbor-backed rollouts; got "
            f"{max_seq_len_int} < {policy_max_new_tokens_int}."
        )

    max_input_tokens = cfg_get(exp_cfg, "model_info", "max_input_tokens")
    max_input_tokens_int = as_positive_int(max_input_tokens)
    if max_input_tokens_int is not None and max_seq_len_int > max_input_tokens_int:
        raise ValueError(
            "runner.trainer.algorithm.max_seq_len must be <= "
            "runner.model_info.max_input_tokens when that limit is set; got "
            f"{max_seq_len_int} > {max_input_tokens_int}."
        )

    async_engine = cfg_get(exp_cfg, "generator", "inference_engine", "async_engine")
    batched = cfg_get(exp_cfg, "generator", "batched")
    if async_engine is True and batched is True:
        raise ValueError(
            "generator.inference_engine.async_engine=true is not supported together "
            "with generator.batched=true for the Harbor generator path."
        )

    colocate_all = cfg_get(exp_cfg, "trainer", "placement", "colocate_all")
    if colocate_all is True:
        policy_num_nodes = as_positive_int(
            cfg_get(exp_cfg, "trainer", "placement", "policy_num_nodes")
        )
        policy_num_gpus_per_node = as_positive_int(
            cfg_get(exp_cfg, "trainer", "placement", "policy_num_gpus_per_node")
        )
        num_engines = as_positive_int(
            cfg_get(exp_cfg, "generator", "inference_engine", "num_engines")
        )
        tensor_parallel_size = as_positive_int(
            cfg_get(exp_cfg, "generator", "inference_engine", "tensor_parallel_size")
        )
        if None not in (
            policy_num_nodes,
            policy_num_gpus_per_node,
            num_engines,
            tensor_parallel_size,
        ):
            assert policy_num_nodes is not None
            assert policy_num_gpus_per_node is not None
            assert num_engines is not None
            assert tensor_parallel_size is not None
            colocated_world = policy_num_nodes * policy_num_gpus_per_node
            inference_world = num_engines * tensor_parallel_size
            if colocated_world != inference_world:
                raise ValueError(
                    "runner.trainer.placement.colocate_all=true requires placement "
                    "and inference topology to match: "
                    "policy_num_nodes * policy_num_gpus_per_node must equal "
                    "generator.inference_engine.num_engines * "
                    f"tensor_parallel_size; got {colocated_world} != {inference_world}."
                )


def build_training_experiment_config(
    cfg: DictConfig | dict | None,
) -> DictConfig:
    """Normalize app-level Hydra config into the distributed experiment shape."""
    root_cfg = to_plain_config(cfg)
    validate_training_controls(root_cfg)
    runner_cfg = root_cfg.get("runner")
    exp_cfg = (
        deepcopy(runner_cfg) if isinstance(runner_cfg, dict) and runner_cfg else deepcopy(root_cfg)
    )

    trainer_cfg = exp_cfg.setdefault("trainer", {})
    if isinstance(trainer_cfg, dict):
        algorithm_cfg = trainer_cfg.setdefault("algorithm", {})
        if isinstance(algorithm_cfg, dict):
            algorithm_cfg.setdefault("advantage_estimator", "tb_log_reward")
            algorithm_cfg.setdefault("advantage_batch_normalize", False)
            policy_loss_type = infer_policy_loss_type(root_cfg, exp_cfg)
            if policy_loss_type is not None:
                algorithm_cfg["policy_loss_type"] = policy_loss_type
            root_objective_cfg = root_cfg.get("objective", {})
            if isinstance(root_objective_cfg, dict):
                promote_objective_algorithm_settings(root_objective_cfg, algorithm_cfg)

        policy_cfg = trainer_cfg.setdefault("policy", {})
        if isinstance(policy_cfg, dict):
            model_cfg = policy_cfg.setdefault("model", {})
            root_policy_cfg = root_cfg.get("policy", {})
            if isinstance(model_cfg, dict) and isinstance(root_policy_cfg, dict):
                model_name = root_policy_cfg.get("model_name")
                if isinstance(model_name, str) and model_name:
                    model_cfg["path"] = model_name

    generator_cfg = exp_cfg.get("generator", {})
    if not isinstance(generator_cfg, dict):
        generator_cfg = {}
        exp_cfg["generator"] = generator_cfg

    inference_engine_cfg = generator_cfg.setdefault("inference_engine", {})
    if not isinstance(inference_engine_cfg, dict):
        inference_engine_cfg = {}
        generator_cfg["inference_engine"] = inference_engine_cfg

    backend = root_cfg.get("backend")
    if isinstance(backend, str) and backend:
        inference_engine_cfg["backend"] = backend

    if isinstance(trainer_cfg, dict):
        policy_cfg = trainer_cfg.get("policy")
        if isinstance(policy_cfg, dict) and isinstance(policy_cfg.get("language_model_only"), bool):
            inference_engine_cfg.setdefault(
                "language_model_only", policy_cfg["language_model_only"]
            )

    if generator_cfg.get("type") == "harbor":
        generator_cfg.setdefault("store_all_messages", True)
        generator_cfg.setdefault("enable_summarize", False)

        data_cfg = exp_cfg.setdefault("data", {})
        if not isinstance(data_cfg, dict):
            data_cfg = {}
            exp_cfg["data"] = data_cfg
        data_cfg.setdefault("val_data", [])

        data_cfg.setdefault("train_data", [])

        engine_kwargs = inference_engine_cfg.get("engine_init_kwargs", {})
        if isinstance(engine_kwargs, dict) and "max_model_len" in engine_kwargs:
            model_info_cfg = exp_cfg.setdefault("model_info", {})
            if isinstance(model_info_cfg, dict):
                model_info_cfg.setdefault("max_input_tokens", engine_kwargs["max_model_len"])

    _translate_root_controls(root_cfg, exp_cfg)
    return OmegaConf.create(exp_cfg)


def _translate_root_controls(root: dict, experiment: dict) -> None:
    """Translate root config controls to their native SkyRL consumers."""
    trainer = experiment.setdefault("trainer", {})
    generator = experiment.setdefault("generator", {})
    policy = trainer.setdefault("policy", {})
    algorithm = trainer.setdefault("algorithm", {})
    if "seed" in root:
        trainer["seed"] = root["seed"]
    for name in ("logZ_init", "logZ_lr"):
        if root.get("objective", {}).get(name) is not None:
            algorithm[name] = root["objective"][name]

    task_overrides: dict[str, object] = {}
    root_env = root.get("env")
    if isinstance(root_env, dict) and root_env:
        env_copy = deepcopy(root_env)
        if (
            isinstance(root.get("seed"), int)
            and not isinstance(root.get("seed"), bool)
            and isinstance(env_copy.get("config"), dict)
        ):
            env_copy["config"] = dict(env_copy["config"])
            env_copy["config"]["evaluation_seed"] = root["seed"]
        task_overrides["env"] = env_copy
    root_judge = root.get("judge")
    if isinstance(root_judge, dict) and root_judge:
        from synthstats.envs.builders import normalize_judge_config

        normalized_judge = normalize_judge_config(root_judge)
        if normalized_judge:
            task_overrides["judge"] = normalized_judge
    if isinstance(root.get("seed"), int) and not isinstance(root.get("seed"), bool):
        task_overrides["seed"] = root["seed"]
    if task_overrides and (
        generator.get("type") == "harbor" or "env" in task_overrides or "judge" in task_overrides
    ):
        existing_overrides = generator.get("task_config_overrides")
        if isinstance(existing_overrides, dict):
            merged_overrides = deepcopy(existing_overrides)
            merged_overrides.update(task_overrides)
            generator["task_config_overrides"] = merged_overrides
        else:
            generator["task_config_overrides"] = task_overrides

    root_policy = root.get("policy", {})
    if "lora" in root_policy:
        policy.setdefault("model", {})["lora"] = root_policy["lora"] or {"rank": 0}
    if root_policy.get("gradient_checkpointing") is not None:
        trainer["gradient_checkpointing"] = root_policy["gradient_checkpointing"]
    sampling = generator.setdefault("sampling_params", {})
    names = {
        "max_new_tokens": "max_generate_length",
        "max_generate_length": "max_generate_length",
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "min_p": "min_p",
        "stop": "stop",
    }
    for name, value in root_policy.get("generation", {}).items():
        if name not in names:
            raise ValueError(f"policy.generation.{name} is not supported by SkyRL")
        sampling[names[name]] = value

    learner = root.get("learner", {})
    for section, values in learner.items():
        if section != "optim" and values:
            raise ValueError(f"learner.{section} is not supported by SkyRL")
    optimizer = policy.setdefault("optimizer_config", {})
    names = {"policy_lr": "lr", "weight_decay": "weight_decay", "max_grad_norm": "max_grad_norm"}
    for name, value in learner.get("optim", {}).items():
        if name == "logZ_lr":
            algorithm[name] = value
        elif name in names:
            optimizer[names[name]] = value
        else:
            raise ValueError(f"learner.optim.{name} is not supported by SkyRL")

    logging_cfg = root.get("logging", {})
    if logging_cfg:
        if logging_cfg.get("entity") is not None:
            raise ValueError("logging.entity is not supported by the SkyRL tracker")
        trainer["logger"] = logging_cfg["backend"]
        if "project" in logging_cfg:
            trainer["project_name"] = logging_cfg["project"]
        if "tags" in logging_cfg:
            trainer["tags"] = logging_cfg["tags"]
    if "name" in root.get("run", {}):
        trainer["run_name"] = root["run"]["name"]
    if logging_cfg.get("name"):
        trainer["run_name"] = logging_cfg["name"]

    checkpoint = root.get("checkpoint", {})
    default_runner_ckpt = (
        f"{root['run']['output_dir']}/checkpoints"
        if isinstance(root.get("run"), dict) and "output_dir" in root["run"]
        else None
    )
    for config_name, skyrl_name in {
        "save_dir": "ckpt_path",
        "every_steps": "ckpt_interval",
        "keep_last": "max_ckpts_to_keep",
    }.items():
        if config_name not in checkpoint:
            continue
        if (
            config_name == "save_dir"
            and checkpoint[config_name] == "checkpoints"
            and isinstance(trainer.get("ckpt_path"), str)
            and trainer["ckpt_path"]
            and trainer["ckpt_path"] not in {"checkpoints", default_runner_ckpt}
        ):
            continue
        trainer[skyrl_name] = checkpoint[config_name]
    if "resume_from" in checkpoint:
        resume = checkpoint["resume_from"]
        trainer["resume_mode"] = "from_path" if resume else "none"
        trainer["resume_path"] = resume


__all__ = [
    "as_positive_int",
    "build_training_experiment_config",
    "cfg_get",
    "infer_policy_loss_type",
    "promote_objective_algorithm_settings",
    "to_plain_config",
    "validate_training_controls",
    "validate_training_experiment_config",
]
