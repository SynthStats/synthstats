"""SDK checks run with the locked SkyRL, ML, PyMC and Harbor extras installed."""

from types import SimpleNamespace

from hydra import compose, initialize_config_module

from synthstats.cli.structured_config import register_train_schemas
from synthstats.integrations.harbor.agent import SynthStatsInstalledAgent
from synthstats.integrations.harbor.generator import SynthStatsHarborGenerator
from synthstats.integrations.skyrl.config import build_native_config
from synthstats.train.experiment_config import build_training_experiment_config


def training_config():
    register_train_schemas()
    with initialize_config_module(config_module="synthstats.configs", version_base=None):
        cfg = compose(config_name="train", overrides=["runner.data.train_data=[train.parquet]"])
    return build_native_config(build_training_experiment_config(cfg))


def test_harbor_generator_implements_the_installed_interface():
    generator = SynthStatsHarborGenerator(
        SimpleNamespace(),
        tokenizer=object(),
        inference_engine_client=SimpleNamespace(get_endpoint_url=lambda: "http://engine.test:1234"),
    )
    assert generator.llm_api_base == "http://engine.test:1234/v1"


def test_native_configuration_contains_backend_defaults_and_public_controls():
    cfg = training_config()
    assert cfg.trainer.strategy == "fsdp"
    assert cfg.trainer.policy.optimizer_config.lr == 1e-5
    assert cfg.generator.sampling_params.max_generate_length == 300
    assert cfg.generator.sampling_params.temperature == 0.7
    assert cfg.trainer.algorithm.policy_loss_type == "gfn_tb"
    assert cfg.trainer.algorithm.logZ_lr == 0.01
    assert cfg.trainer.policy.fsdp_config is not None
    assert cfg.data.train_data == ["train.parquet"]


def test_installed_harbor_agent_is_concrete(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_BASE", "http://engine.test:1234/v1")
    agent = SynthStatsInstalledAgent(logs_dir=tmp_path, model_name="Qwen/Qwen3.5-0.8B")
    assert agent.name() == "synthstats"


def test_installed_harbor_trial_factory_constructs_the_concrete_task(tmp_path):
    import asyncio

    from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
    from harbor.trial.trial import Trial

    from synthstats.integrations.harbor.task_registry import resolve_task_package_path

    config = TrialConfig(
        task=TaskConfig(path=resolve_task_package_path("dugongs")),
        agent=AgentConfig(
            import_path="synthstats.integrations.harbor.agent:SynthStatsInstalledAgent",
            env={"LLM_API_BASE": "http://engine.test/v1"},
        ),
        trials_dir=tmp_path,
    )
    trial = asyncio.run(Trial.create(config))
    assert trial.agent.llm_base_url == "http://engine.test/v1"
    assert trial.task.name == "dugongs"


def test_episode_atif_validates_against_installed_harbor_trajectory_schema(tmp_path, monkeypatch):
    import asyncio
    import json

    import httpx
    from harbor.models.trajectories.trajectory import Trajectory

    import synthstats.integrations.harbor.episode as episode_mod
    from synthstats.integrations.harbor.episode import run_episode

    class _FakeEnv:
        def __init__(self):
            self.chat_history = []

        def init(self):
            self.chat_history = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "obs0"},
            ]
            return list(self.chat_history), {}

        def step(self, action: str):
            return {
                "done": True,
                "reward": 1.0,
                "metadata": {"program": "x = 1", "language": "pymc"},
            }

    monkeypatch.setattr(episode_mod, "build_env_from_dict", lambda cfg: _FakeEnv())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [10, 11]})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "text": "<submit>x = 1</submit>",
                        "token_ids": [20, 21],
                        "finish_reason": "stop",
                        "logprobs": {"token_logprobs": [-0.1, -0.2]},
                    }
                ]
            },
        )

    orig_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        episode_mod.httpx,
        "AsyncClient",
        lambda **kwargs: orig_async_client(
            base_url=kwargs.get("base_url", "http://endpoint.test"),
            transport=httpx.MockTransport(handler),
        ),
    )

    runtime = {
        "TASK_CONFIG": json.dumps({"env": {"name": "dugongs"}}),
        "TASK_PROVENANCE": json.dumps({"task_name": "dugongs"}),
        "LLM_API_BASE": "http://endpoint.test/v1",
        "MODEL_NAME": "Qwen/Qwen3.5-0.8B",
        "SAMPLING_PARAMS": json.dumps({"max_generate_length": 16, "temperature": 0.7}),
        "MAX_SEQUENCE_LENGTH": "128",
        "MAX_TURNS": "2",
    }
    asyncio.run(run_episode("Fit dugongs.", runtime, tmp_path))
    atif = json.loads((tmp_path / "trajectory.json").read_text())
    validated = Trajectory.model_validate(atif)
    assert validated.schema_version == "ATIF-v1.6"


def test_gfn_policy_worker_registers_steps_and_restores_log_z(monkeypatch):
    import copy

    import pytest
    import torch
    import torch.nn as nn
    from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase

    from synthstats.integrations.skyrl.policy_worker import GFNPolicyWorker
    from synthstats.train.objectives.gfn_losses import gfn_tb_loss

    class _ToyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_bias = nn.Parameter(torch.tensor([[-0.5, -0.5]], dtype=torch.float32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.logit_bias + x

    def _fake_base_init_model(self, model_path, num_training_steps=None):
        inner = _ToyBackbone()
        self._seq_parallel_monkey_patch(inner)
        self.model = SimpleNamespace(model=inner)
        self.optimizer = torch.optim.AdamW(
            inner.parameters(),
            lr=self.cfg.policy.optimizer_config.lr,
            weight_decay=0.0,
        )
        self.scheduler = None

    monkeypatch.setattr(FSDPPolicyWorkerBase, "init_model", _fake_base_init_model)
    monkeypatch.setattr(
        FSDPPolicyWorkerBase,
        "_seq_parallel_monkey_patch",
        lambda self, model: None,
    )

    def _make_worker(logz_init: float) -> GFNPolicyWorker:
        worker = GFNPolicyWorker.__new__(GFNPolicyWorker)
        worker.cfg = SimpleNamespace(
            policy=SimpleNamespace(
                inference_only_init=False,
                optimizer_config=SimpleNamespace(
                    lr=1e-4,
                    scheduler="constant",
                    num_warmup_steps=0,
                ),
            ),
            algorithm=SimpleNamespace(
                logZ_init=logz_init,
                logZ_lr=0.05,
                tb_max_residual=100.0,
            ),
        )
        worker.init_model("dummy", num_training_steps=10)
        return worker

    worker = _make_worker(logz_init=1.25)
    inner = worker.model.model
    assert "log_z" in dict(inner.named_parameters())
    assert len(worker.optimizer.param_groups) == 2
    assert worker.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert worker.optimizer.param_groups[1]["lr"] == pytest.approx(0.05)
    assert worker.optimizer.param_groups[1]["params"] == [inner.log_z]

    x = torch.zeros((1, 2), dtype=torch.float32)
    advantages = torch.full((1, 2), 2.5, dtype=torch.float32)
    mask = torch.ones((1, 2), dtype=torch.float32)

    log_probs = inner(x)
    loss, metrics = gfn_tb_loss(
        log_probs,
        log_probs.detach(),
        advantages,
        worker.cfg.algorithm,
        loss_mask=mask,
    )
    assert metrics["logZ"] == pytest.approx(1.25)
    assert metrics["logZ_lr"] == pytest.approx(0.05)

    loss.backward()
    assert inner.log_z.grad is not None
    assert inner.log_z.grad.abs().item() > 0.0
    assert inner.logit_bias.grad is not None

    before_bias = inner.logit_bias.detach().clone()
    before_logz = inner.log_z.detach().clone()
    worker.optimizer.step()
    worker.optimizer.zero_grad()

    bias_step = (inner.logit_bias.detach() - before_bias).abs().max().item()
    logz_step = (inner.log_z.detach() - before_logz).abs().item()
    assert bias_step == pytest.approx(1e-4, rel=1e-2)
    assert logz_step == pytest.approx(0.05, rel=1e-2)

    saved_model = copy.deepcopy(inner.state_dict())
    saved_optim = copy.deepcopy(worker.optimizer.state_dict())

    restored = _make_worker(logz_init=0.0)
    restored.model.model.load_state_dict(saved_model)
    restored.optimizer.load_state_dict(saved_optim)
    assert restored.model.model.log_z.item() == pytest.approx(inner.log_z.item())

    for current in (worker, restored):
        lp = current.model.model(x)
        step_loss, _ = gfn_tb_loss(
            lp,
            lp.detach(),
            advantages,
            current.cfg.algorithm,
            loss_mask=mask,
        )
        step_loss.backward()
        current.optimizer.step()
        current.optimizer.zero_grad()

    assert restored.model.model.log_z.item() == pytest.approx(inner.log_z.item())
    assert torch.allclose(restored.model.model.logit_bias, inner.logit_bias)

    import math

    from skyrl.backends.skyrl_train.utils.ppo_utils import compute_advantages_and_returns

    native_cfg = training_config()
    token_rewards = torch.tensor([[0.0, 2.0], [0.0, 4.0]], dtype=torch.float32)
    response_mask = torch.ones((2, 2), dtype=torch.float32)
    computed_adv, _ = compute_advantages_and_returns(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=["0", "0"],
        adv_estimator=native_cfg.trainer.algorithm.advantage_estimator,
        config=native_cfg.trainer.algorithm,
        values=None,
        gamma=1.0,
        lambd=1.0,
        grpo_norm_by_std=True,
    )
    if native_cfg.trainer.algorithm.advantage_batch_normalize:
        mean = computed_adv.mean()
        std = computed_adv.std()
        computed_adv = (computed_adv - mean) / (std + 1e-8)

    expected_log_r = torch.tensor([math.log(2.0), math.log(4.0)], dtype=torch.float32)
    torch.testing.assert_close(
        computed_adv,
        expected_log_r.unsqueeze(1).expand(2, 2),
    )
    batch_lp = worker.model.model(torch.zeros((2, 2), dtype=torch.float32))
    batch_loss, _ = gfn_tb_loss(
        batch_lp,
        batch_lp.detach(),
        computed_adv,
        worker.cfg.algorithm,
        loss_mask=response_mask,
    )
    expected_residuals = (
        worker.model.model.log_z.detach() + batch_lp.detach().sum(dim=-1) - expected_log_r
    )
    assert batch_loss.item() == pytest.approx((expected_residuals**2).mean().item(), rel=1e-6)


def test_ci_contracts_job_supplies_container_image_prerequisite(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    import pytest
    import yaml

    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "test_container_isolation",
        Path(__file__).resolve().parent / "test_container_isolation.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ci_workflow = yaml.safe_load(
        (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    contract_steps = ci_workflow["jobs"]["contracts"]["steps"]
    pytest_step = next(
        step for step in contract_steps if "pytest checks/" in str(step.get("run", ""))
    )
    image_name = (pytest_step.get("env") or {}).get("SYNTHSTATS_TEST_IMAGE")
    assert isinstance(image_name, str) and image_name.strip()
    assert any(
        "docker build" in str(step.get("run", "")) and image_name in str(step.get("run", ""))
        for step in contract_steps
    )

    monkeypatch.delenv("SYNTHSTATS_TEST_IMAGE", raising=False)
    with pytest.raises(pytest.skip.Exception, match="SYNTHSTATS_TEST_IMAGE"):
        module.test_program_cannot_write_the_controllers_mounted_directory(tmp_path)
