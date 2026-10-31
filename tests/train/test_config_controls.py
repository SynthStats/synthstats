import pytest
from omegaconf import OmegaConf

from synthstats.train.experiment_config import build_training_experiment_config


def test_config_controls_reach_the_training_backend():
    root = {
        "seed": 17,
        "backend": "vllm",
        "policy": {
            "model_name": "Qwen/Qwen3.5-0.8B",
            "generation": {"temperature": 0.35, "top_p": 0.8, "max_new_tokens": 71},
            "lora": {"rank": 8, "alpha": 16, "target_modules": ["q_proj"]},
            "gradient_checkpointing": True,
        },
        "objective": {"loss_type": "tb", "logZ_init": 1.5, "logZ_lr": 0.03},
        "learner": {"optim": {"policy_lr": 0.0002, "weight_decay": 0.04, "max_grad_norm": 0.6}},
        "logging": {"backend": "wandb", "project": "comparison", "tags": ["trial"]},
        "checkpoint": {
            "save_dir": "/tmp/saved",
            "every_steps": 7,
            "keep_last": 2,
            "resume_from": "/tmp/previous",
        },
        "run": {"name": "trial-3"},
        "runner": {"trainer": {}, "generator": {}, "data": {"train_data": ["train.parquet"]}},
    }
    cfg = build_training_experiment_config(OmegaConf.create(root))
    assert cfg.trainer.seed == 17
    assert cfg.trainer.policy.optimizer_config == {
        "lr": 0.0002,
        "weight_decay": 0.04,
        "max_grad_norm": 0.6,
    }
    assert cfg.generator.sampling_params == {
        "temperature": 0.35,
        "top_p": 0.8,
        "max_generate_length": 71,
    }
    assert cfg.trainer.policy.model.lora.rank == 8
    assert cfg.trainer.gradient_checkpointing is True
    assert cfg.trainer.algorithm.logZ_init == 1.5
    assert cfg.trainer.algorithm.logZ_lr == 0.03
    assert cfg.trainer.logger == "wandb"
    assert cfg.trainer.project_name == "comparison"
    assert cfg.trainer.run_name == "trial-3"
    assert cfg.trainer.ckpt_path == "/tmp/saved"
    assert cfg.trainer.ckpt_interval == 7
    assert cfg.trainer.max_ckpts_to_keep == 2
    assert cfg.trainer.resume_mode == "from_path"
    assert cfg.trainer.resume_path == "/tmp/previous"


def test_config_normalization_preserves_explicit_parquet_inputs():
    cfg = build_training_experiment_config(
        {
            "runner": {
                "generator": {"type": "harbor"},
                "data": {"train_data": ["a.parquet"], "val_data": ["b.parquet"]},
            },
        }
    )
    assert cfg.data.train_data == ["a.parquet"]
    assert cfg.data.val_data == ["b.parquet"]


def test_seed_controls_bundled_task_observations():
    from hydra import compose, initialize_config_module

    from synthstats.cli.structured_config import register_train_schemas
    from synthstats.envs.builders import build_env_from_dict
    from synthstats.integrations.harbor.generator import normalize_task_request
    from synthstats.integrations.harbor.task_registry import build_local_task_request

    register_train_schemas()
    with initialize_config_module(config_module="synthstats.configs", version_base=None):
        cfg = compose(config_name="train", overrides=["seed=17"])
    exp_cfg = build_training_experiment_config(cfg)
    base_request = build_local_task_request({"env": {"name": "dugongs"}})
    normalized = normalize_task_request(
        base_request,
        overrides=OmegaConf.to_container(exp_cfg.generator.task_config_overrides, resolve=True),
        default_seed=int(exp_cfg.trainer.seed),
    )
    first = build_env_from_dict(normalized.task_config)
    second = build_env_from_dict(normalized.task_config)
    first.init()
    second.init()
    assert first._random_seed == 17
    assert second._random_seed == 17
    assert [first.task.env.query(f"age={age}") for age in range(1, 9)] == [
        second.task.env.query(f"age={age}") for age in range(1, 9)
    ]


@pytest.mark.parametrize("control", ["precision", "tinker"])
def test_unimplemented_learner_controls_raise(control):
    with pytest.raises(ValueError, match=f"learner.{control}"):
        build_training_experiment_config({"learner": {control: {"enabled": True}}})


def test_explicit_logging_name_reaches_the_tracker():
    cfg = build_training_experiment_config(
        {
            "logging": {"backend": "console", "name": "explicit-name"},
            "run": {"name": "directory-name"},
        }
    )
    assert cfg.trainer.run_name == "explicit-name"


def test_explicit_runner_ckpt_path_is_not_clobbered_by_default_checkpoint_save_dir():
    from hydra import compose, initialize_config_module

    from synthstats.cli.structured_config import register_train_schemas

    register_train_schemas()
    with initialize_config_module(config_module="synthstats.configs", version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["runner.trainer.ckpt_path=/tmp/requested-by-runner"],
        )
    exp_cfg = build_training_experiment_config(cfg)
    assert exp_cfg.trainer.ckpt_path == "/tmp/requested-by-runner"


def test_launch_training_script_rejects_tinker_and_forwards_checkpoint_save_dir(tmp_path):
    import os
    import subprocess
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "launch-training.sh"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured_args = tmp_path / "uv_args.txt"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "{captured_args}"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    bad = subprocess.run(
        ["bash", str(script_path)],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
            "BACKEND": "tinker",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode != 0
    assert "Unknown backend: tinker" in bad.stderr
    assert "tinker" not in bad.stderr.split("Options:")[-1]
    assert not captured_args.exists()

    good = subprocess.run(
        ["bash", str(script_path)],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
            "BACKEND": "vllm",
            "CKPT_PATH": "/tmp/custom-ckpts",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert good.returncode == 0
    args = captured_args.read_text(encoding="utf-8").splitlines()
    assert "checkpoint.save_dir=/tmp/custom-ckpts" in args
