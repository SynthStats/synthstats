from pathlib import Path
from typing import Any, cast

import pytest
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from synthstats.cli.structured_config import register_train_schemas
from synthstats.cli.train import _bootstrap_environment, build_runner, render_debug_config
from synthstats.integrations.harbor.task_registry import build_local_task_request
from synthstats.train.experiment_config import build_training_experiment_config
from synthstats.train.runners.base import RunResult

CONFIG_DIR = Path(__file__).resolve().parents[2] / "src" / "synthstats" / "configs"


def _compose_train_config(*, overrides: list[str] | None = None):
    from hydra import compose, initialize_config_dir

    register_train_schemas()
    with initialize_config_dir(config_dir=str(CONFIG_DIR.absolute()), version_base=None):
        return compose(config_name="train", overrides=overrides or [])


def _mock_skyrl_available(monkeypatch):
    import synthstats.integrations.skyrl.runner as skyrl_runner_module

    monkeypatch.setattr(skyrl_runner_module, "SKYRL_TRAINER_AVAILABLE", True)
    monkeypatch.setattr(skyrl_runner_module, "_BasePPOExp", object())
    monkeypatch.setattr(skyrl_runner_module, "_initialize_ray", object())
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(
        sys.modules,
        "synthstats.integrations.skyrl.config",
        SimpleNamespace(build_native_config=lambda cfg: cfg),
    )
    return skyrl_runner_module


class TestGetRunner:
    def test_skyrl_runner_dispatches_when_available(self, monkeypatch):
        skyrl_runner_module = _mock_skyrl_available(monkeypatch)

        cfg = OmegaConf.create(
            {
                "runner": {"type": "skyrl"},
                "objective": {"loss_type": "tb"},
                "seed": 42,
                "device": "cpu",
            }
        )
        runner = build_runner(cfg)

        assert isinstance(runner, skyrl_runner_module.SkyRLRunner)

    def test_skyrl_harbor_alias_dispatches_to_skyrl(self, monkeypatch):
        skyrl_runner_module = _mock_skyrl_available(monkeypatch)
        cfg = OmegaConf.create(
            {
                "runner": {"type": "skyrl_harbor"},
                "objective": {"loss_type": "tb"},
                "seed": 42,
                "device": "cpu",
            }
        )
        runner = build_runner(cfg)

        assert isinstance(runner, skyrl_runner_module.SkyRLRunner)

    def test_default_is_skyrl(self, monkeypatch):
        skyrl_runner_module = _mock_skyrl_available(monkeypatch)
        cfg = OmegaConf.create(
            {
                "runner": {},
                "objective": {"loss_type": "tb"},
                "seed": 42,
                "device": "cpu",
            }
        )
        runner = build_runner(cfg)

        assert isinstance(runner, skyrl_runner_module.SkyRLRunner)

    @pytest.mark.parametrize("runner_type", ["local", "tinker", "harbor", "runpod"])
    def test_invalid_runner_type_raises(self, runner_type: str):
        cfg = OmegaConf.create(
            {
                "runner": {"type": runner_type},
                "seed": 42,
                "device": "cpu",
            }
        )
        with pytest.raises(
            ValueError,
            match=f"runner={runner_type} is not a valid runner type",
        ):
            build_runner(cfg)

    def test_unknown_runner_raises(self):
        cfg = OmegaConf.create(
            {
                "runner": {"type": "unknown"},
            }
        )
        with pytest.raises(ValueError, match="Valid options: skyrl, skyrl_harbor"):
            build_runner(cfg)

    @pytest.mark.parametrize("backend", ["local", "runpod", "unknown"])
    def test_invalid_backend_raises(self, backend: str):
        cfg = OmegaConf.create(
            {
                "runner": {"type": "skyrl"},
                "backend": backend,
                "objective": {"loss_type": "subtb"},
                "policy": {"model_name": "Qwen/Qwen3.5-0.8B"},
            }
        )
        with pytest.raises(ValueError, match=f"backend={backend} is not a valid backend"):
            build_runner(cfg)

    @pytest.mark.parametrize("loss_type", ["modified_subtb", "ab_subtb", "agentic_subtb"])
    def test_experimental_loss_types_raise(self, loss_type: str):
        cfg = OmegaConf.create(
            {
                "runner": {"type": "skyrl"},
                "backend": "vllm",
                "objective": {"loss_type": loss_type},
                "policy": {"model_name": "Qwen/Qwen3.5-0.8B"},
            }
        )
        with pytest.raises(
            ValueError,
            match=f"objective.loss_type={loss_type} is not supported by the training CLI",
        ):
            build_runner(cfg)

    def test_policy_backend_mismatch_raises(self):
        cfg = OmegaConf.create(
            {
                "runner": {"type": "skyrl"},
                "backend": "vllm",
                "objective": {"loss_type": "tb"},
                "policy": {
                    "model_name": "Qwen/Qwen3.5-4B",
                    "allowed_backends": ["tinker"],
                },
            }
        )
        with pytest.raises(
            ValueError,
            match="does not support backend=vllm",
        ):
            build_runner(cfg)


class TestRunnerIntegration:
    def test_default_train_config_builds_skyrl_runner(self, monkeypatch):
        skyrl_runner_module = _mock_skyrl_available(monkeypatch)
        cfg = _compose_train_config()

        runner = build_runner(cfg)

        assert isinstance(runner, skyrl_runner_module.SkyRLRunner)
        assert runner.exp_cfg.generator.type == "harbor"
        assert (
            runner.exp_cfg.generator.agent_import_path
            == "synthstats.integrations.harbor.agent:SynthStatsInstalledAgent"
        )
        assert runner.exp_cfg.generator.inference_engine.backend == "vllm"
        assert runner.exp_cfg.generator.llm_api_base is None
        assert runner.exp_cfg.data.train_data == []

    def test_default_train_config_preserves_env_owned_judge_in_task_request(self):
        cfg = _compose_train_config()

        request = build_local_task_request(cfg)

        assert "judge" not in request["task_config"]
        assert (
            request["task_config"]["env"]["judge"]["_target_"]
            == "synthstats.judges.likelihood.LikelihoodJudge"
        )
        assert request["task_config"]["env"]["judge"]["loo_gate_mode"] == "strict"

    def test_lazyppl_train_config_preserves_env_owned_composite_judge_in_task_request(
        self,
    ):
        cfg = _compose_train_config(overrides=["env=lazyppl_dugongs"])

        request = build_local_task_request(cfg)

        assert "judge" not in request["task_config"]
        assert (
            request["task_config"]["env"]["judge"]["_target_"]
            == "synthstats.judges.composite.CompositeJudge"
        )
        assert len(request["task_config"]["env"]["judge"]["judges"]) == 3

    def test_explicit_judge_override_flows_into_task_request(self):
        cfg = _compose_train_config(overrides=["judge=hybrid_lm"])

        request = build_local_task_request(cfg)

        assert request["task_config"]["judge"]["_target_"] == (
            "synthstats.judges.composite.CompositeJudge"
        )

    def test_tinker_backend_refuses_unimplemented_skyrl_inference(self):
        cfg = _compose_train_config(overrides=["backend=tinker", "policy=tinker_qwen3_5_4b"])
        with pytest.raises(ValueError, match="backend=tinker is not a valid backend"):
            build_runner(cfg)

    def test_render_debug_config_redacts_secrets(self):
        cfg = OmegaConf.create(
            {
                "policy": {"model_name": "Qwen/Qwen3.5-0.8B"},
                "generator": {"llm_api_key": "fake-llm-key-1"},
            }
        )

        rendered = render_debug_config(cfg)

        assert "***REDACTED***" in rendered
        assert "fake-llm-key-1" not in rendered

    def test_bootstrap_environment_runs_before_oc_env_composition(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / ".env").write_text("SYNTHSTATS_JUDGE_API_BASE=from-dotenv\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("SYNTHSTATS_JUDGE_API_BASE", raising=False)

        _bootstrap_environment()
        cfg = _compose_train_config(overrides=["judge=llm_critic_glm5"])

        assert cfg.judge.config.api_base == "from-dotenv"

    def test_bootstrap_environment_preserves_existing_env_values(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / ".env").write_text("SYNTHSTATS_JUDGE_API_BASE=from-dotenv\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SYNTHSTATS_JUDGE_API_BASE", "from-environment")

        _bootstrap_environment()
        cfg = _compose_train_config(overrides=["judge=llm_critic_glm5"])

        assert cfg.judge.config.api_base == "from-environment"


class TestConfigFiles:
    @pytest.fixture
    def config_dir(self):
        return CONFIG_DIR

    def test_train_config_exists(self, config_dir):
        train_config = config_dir / "train.yaml"
        if not train_config.exists():
            pytest.skip("train.yaml not found")
        cfg = OmegaConf.load(train_config)
        assert "defaults" in cfg

    def test_train_config_sets_explicit_hydra_job_chdir(self, config_dir):
        cfg = OmegaConf.load(config_dir / "train.yaml")
        assert cfg.hydra.job.chdir is False

    def test_runner_configs_exist(self, config_dir):
        runner_dir = config_dir / "runner"
        if not runner_dir.exists():
            pytest.skip("runner configs not found")
        assert (runner_dir / "skyrl.yaml").exists()
        assert (runner_dir / "skyrl_harbor.yaml").exists()

    def test_runner_configs_use_run_scoped_artifact_paths(self, config_dir):
        runner_dir = config_dir / "runner"
        for name in ("skyrl.yaml", "skyrl_harbor.yaml"):
            cfg = OmegaConf.load(runner_dir / name)
            trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=False)
            assert isinstance(trainer_cfg, dict)
            trainer_cfg = cast(dict[str, Any], trainer_cfg)
            assert trainer_cfg["ckpt_path"] == "${run.output_dir}/checkpoints"
            assert trainer_cfg["export_path"] == "${run.output_dir}/export"

    def test_harbor_runner_config_sets_explicit_llm_api_base(self, config_dir):
        cfg = OmegaConf.load(config_dir / "runner" / "skyrl_harbor.yaml")
        assert cfg.generator.llm_api_base is None

    def test_hybrid_judge_config_uses_strict_loo_gate(self, config_dir):
        cfg = OmegaConf.load(config_dir / "judge" / "hybrid_lm.yaml")
        assert cfg.judges[0].judge._target_ == "synthstats.judges.likelihood.LikelihoodJudge"
        assert cfg.judges[0].judge.loo_gate_mode == "strict"

    def test_likelihood_judge_config_uses_strict_loo_gate(self, config_dir):
        cfg = OmegaConf.load(config_dir / "judge" / "likelihood.yaml")
        assert cfg._target_ == "synthstats.judges.likelihood.LikelihoodJudge"
        assert cfg.loo_gate_mode == "strict"

    def test_backend_configs_exist(self, config_dir):
        backend_dir = config_dir / "backend"
        assert backend_dir.exists()
        assert (backend_dir / "vllm.yaml").exists()
        assert (backend_dir / "tinker.yaml").exists()


class TestRunResult:
    def test_run_result_defaults(self):
        result = RunResult()
        assert result.metrics == {}
        assert result.checkpoints == []
        assert result.error is None
        assert result.interrupted is False

    def test_run_result_with_metrics(self):
        result = RunResult(
            metrics={"loss": 0.5, "logZ": 1.2},
            checkpoints=["/path/to/ckpt.pt"],
        )
        assert result.metrics["loss"] == 0.5
        assert len(result.checkpoints) == 1

    def test_run_result_with_error(self):
        result = RunResult(error="Training failed")
        assert result.error == "Training failed"


class TestHydraConfigIntegration:
    @pytest.fixture
    def config_dir(self):
        return CONFIG_DIR

    def test_train_config_loads(self, config_dir):
        if not config_dir.exists():
            pytest.skip("Config directory not found")

        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None):
            cfg = compose(config_name="train")
            assert cfg.runner.type == "skyrl"
            assert cfg.runner.generator.type == "harbor"
            assert cfg.backend == "vllm"
            assert cfg.runner.generator.inference_engine.backend == "vllm"
            assert cfg.env.name == "dugongs"
            assert "policy" in cfg

    def test_model_override(self, config_dir):
        if not (config_dir / "policy" / "hf_qwen3_5_4b.yaml").exists():
            pytest.skip("Policy configs not found")

        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None):
            cfg = compose(config_name="train", overrides=["policy=hf_qwen3_5_4b"])
            assert cfg.policy.model_name == "Qwen/Qwen3.5-4B"

    def test_tinker_backend_override(self, config_dir):
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None):
            cfg = compose(
                config_name="train",
                overrides=["backend=tinker", "policy=tinker_qwen3_5_4b"],
            )
            assert cfg.backend == "tinker"
            assert cfg.policy.model_name == "Qwen/Qwen3.5-4B"
            assert list(cfg.policy.allowed_backends) == ["tinker"]

    def test_default_train_config_uses_env_native_strict_loo_gate(self, config_dir):
        from hydra import compose, initialize_config_dir

        register_train_schemas()
        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None):
            cfg = compose(config_name="train")
            assert cfg.env.judge._target_ == "synthstats.judges.likelihood.LikelihoodJudge"
            assert cfg.env.judge.loo_gate_mode == "strict"
            assert cfg.judge._target_ is None

    def test_runner_schema_rejects_unknown_override_key(self, config_dir):
        from hydra import compose, initialize_config_dir

        register_train_schemas()
        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None):
            with pytest.raises(ConfigCompositionException, match="not_a_real_field"):
                compose(config_name="train", overrides=["runner.not_a_real_field=1"])

    def test_train_schema_rejects_unknown_top_level_key(self, tmp_path: Path):
        from hydra import compose, initialize_config_dir

        register_train_schemas()
        (tmp_path / "train.yaml").write_text(
            "\n".join(
                [
                    "defaults:",
                    "  - train_schema",
                    "  - _self_",
                    "unexpected_key: boom",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        with initialize_config_dir(config_dir=str(tmp_path), version_base=None):
            with pytest.raises(
                ConfigCompositionException,
                match="unexpected_key|TrainSurfaceConfig",
            ):
                compose(config_name="train")

    def test_every_shipped_config_composes_cleanly(self, config_dir):
        from hydra import compose, initialize_config_dir

        register_train_schemas()
        default_groups = {
            "backend",
            "checkpoint",
            "env",
            "hydra/launcher",
            "judge",
            "learner",
            "logging",
            "objective",
            "policy",
            "runner",
        }
        yaml_paths = sorted(config_dir.rglob("*.yaml"))
        assert yaml_paths

        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None):
            for path in yaml_paths:
                rel = path.relative_to(config_dir)
                if rel == Path("train.yaml"):
                    overrides: list[str] = []
                else:
                    group = str(rel.parent).replace("\\", "/")
                    prefix = "" if group in default_groups else "+"
                    overrides = [f"{prefix}{group}={path.stem}"]
                cfg = compose(config_name="train", overrides=overrides)
                assert cfg.runner.type in {"skyrl", "skyrl_harbor"}, f"Failed for {rel}"


@pytest.mark.parametrize(
    ("overrides", "objective", "policy_loss"),
    [([], "tb", "gfn_tb"), (["objective.loss_type=subtb"], "subtb", "gfn_subtb")],
)
def test_composed_training_defaults_to_tb_and_preserves_subtb(overrides, objective, policy_loss):
    cfg = _compose_train_config(overrides=overrides)
    exp_cfg = build_training_experiment_config(cfg)
    assert cfg.objective.loss_type == objective
    assert exp_cfg.trainer.algorithm.policy_loss_type == policy_loss
