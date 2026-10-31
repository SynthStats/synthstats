"""Focused tests for Harbor generator reasoning-token semantics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import synthstats.integrations.harbor.generator as generator_module
from synthstats.integrations.harbor.generator import (
    _unavailable_generator,
    build_task_provenance,
    empty_rollout_lineage_flags,
    empty_rollout_lineage_metrics,
    empty_rollout_metrics,
    ensure_local_task_image,
    extract_generation_data_from_agent_metadata,
    extract_generation_data_from_atif,
    extract_lineage_artifacts_from_atif,
    extract_reward_from_trial_result,
    extract_reward_signal_from_trial_result,
    extract_rollout_lineage_flags,
    normalize_task_request,
    record_rollout_lineage_metrics,
    resolve_llm_api_base,
    resolve_local_repo_mount_config,
    should_delete_trial_environment,
)
from synthstats.integrations.harbor.task_registry import (
    REGISTERED_TASK_PACKAGES,
    TASK_PACKAGES_DIR,
    build_local_task_request,
    resolve_task_package_path,
)


class _CharTokenizer:
    def __init__(self, chars: str):
        self.vocab = {c: i for i, c in enumerate(chars)}
        self.reverse_vocab = {i: c for c, i in self.vocab.items()}

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(self.reverse_vocab.get(tid, "") for tid in token_ids)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self.vocab[c] for c in text if c in self.vocab]


def _build_test_generator(monkeypatch: pytest.MonkeyPatch):
    class _BaseGenerator:
        def __init__(self, *args, **kwargs):
            return None

    monkeypatch.setattr(generator_module, "_GeneratorInterface", _BaseGenerator)
    monkeypatch.setattr(generator_module, "_HARBOR_AVAILABLE", True)
    generator_cls = generator_module._make_generator_class()
    assert generator_cls is not None
    return generator_cls(config=object(), llm_api_base="http://engine.test/v1")


def test_extract_generation_data_keeps_reasoning_tokens_trainable():
    tokenizer = _CharTokenizer("abcdefghijklmnopqrstuvwxyz<>/! ")
    reasoning = "abc"
    message = "xyz"
    full_text = f"<think>{reasoning}</think>{message}"
    token_ids = tokenizer.encode(full_text)
    logprobs = [-0.1] * len(token_ids)

    data = extract_generation_data_from_atif(
        {
            "steps": [
                {
                    "source": "agent",
                    "message": message,
                    "reasoning_content": reasoning,
                    "metrics": {
                        "prompt_token_ids": [100, 101],
                        "completion_token_ids": token_ids,
                        "logprobs": logprobs,
                    },
                }
            ]
        }
    )

    assert data is not None
    assert data["loss_mask"] == [1] * len(token_ids)
    assert data["logprobs"] == logprobs


def test_extract_generation_data_from_agent_metadata_prefers_explicit_masks():
    data = extract_generation_data_from_agent_metadata(
        {
            "prompt_ids": [1, 2],
            "token_ids": [3, 4, 5],
            "mask_ids": [1, 0, 1],
            "logprobs": [-0.1, -0.2, -0.3],
            "stop_reason": "submit_program",
        }
    )

    assert data == {
        "prompt_ids": [1, 2],
        "response_ids": [3, 4, 5],
        "logprobs": [-0.1, -0.2, -0.3],
        "loss_mask": [1, 0, 1],
        "stop_reason": "submit_program",
    }


def test_extract_atif_supports_dict_shaped_trial_result(monkeypatch):
    generator = _build_test_generator(monkeypatch)
    trajectory = {"schema_version": "ATIF-v1.5", "steps": [{"source": "agent"}]}

    result = generator._extract_atif({"agent_result": {"trajectory": trajectory}})

    assert result == trajectory


def test_extract_generation_data_supports_dict_shaped_trial_result(monkeypatch):
    generator = _build_test_generator(monkeypatch)

    result = generator._extract_generation_data(
        {
            "agent_result": {
                "metadata": {
                    "prompt_ids": [1, 2],
                    "token_ids": [3, 4],
                    "mask_ids": [1, 1],
                    "logprobs": [-0.1, -0.2],
                    "stop_reason": "submit_program",
                }
            }
        },
        atif={"schema_version": "ATIF-v1.5", "steps": []},
    )

    assert result == {
        "prompt_ids": [1, 2],
        "response_ids": [3, 4],
        "logprobs": [-0.1, -0.2],
        "loss_mask": [1, 1],
        "stop_reason": "submit_program",
    }


def test_normalize_task_request_separates_task_package_path_from_env_payload():
    normalized = normalize_task_request(
        {
            "task_name": "dugongs",
            "task_path": "/tmp/dugongs",
            "task_config": {"env": {"name": "dugongs"}},
            "source": "local_task_package",
        }
    )

    assert normalized.task_kwargs == {
        "path": "/tmp/dugongs",
        "source": "local_task_package",
    }
    assert normalized.task_config == {"env": {"name": "dugongs"}}
    assert normalized.task_name == "dugongs"


def test_build_task_provenance_uses_task_ref_fields():
    normalized = normalize_task_request(
        {
            "task_name": "dugongs",
            "task_path": "/tmp/dugongs",
            "task_config": {"env": {"name": "dugongs"}},
            "source": "local_task_package",
            "git_commit_id": "abc123",
        }
    )

    assert build_task_provenance(normalized) == {
        "task_name": "dugongs",
        "path": "/tmp/dugongs",
        "source": "local_task_package",
        "git_commit_id": "abc123",
    }


def test_resolve_local_repo_mount_config_for_packaged_task():
    mounts, env = resolve_local_repo_mount_config({"path": str(TASK_PACKAGES_DIR / "dugongs")})

    assert mounts is not None
    assert mounts[0]["target"] == "/workspace/synthstats"
    assert env == {"PYTHONPATH": "/workspace/synthstats/src"}


def test_resolve_local_repo_mount_config_skips_external_paths():
    mounts, env = resolve_local_repo_mount_config({"path": "/tmp/dugongs"})

    assert mounts is None
    assert env == {}


def test_should_delete_trial_environment_keeps_local_task_images() -> None:
    normalized = normalize_task_request(
        {
            "task_name": "dugongs",
            "task_path": "/tmp/synthstats-harbor-dugongs-prebuilt",
            "task_config": {"env": {"name": "dugongs"}},
        }
    )

    assert should_delete_trial_environment(normalized) is False


def test_should_delete_trial_environment_cleans_up_external_tasks() -> None:
    normalized = normalize_task_request(
        {
            "task_path": "/tmp/external-task",
            "task_config": {"env": {"name": "external"}},
        }
    )

    assert should_delete_trial_environment(normalized) is True


def test_ensure_local_task_image_is_noop_for_external_tasks(monkeypatch) -> None:
    normalized = normalize_task_request(
        {
            "task_path": "/tmp/external-task",
            "task_config": {"env": {"name": "external"}},
        }
    )

    def _unexpected_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called for external tasks")

    monkeypatch.setattr("synthstats.integrations.harbor.generator.subprocess.run", _unexpected_run)

    ensure_local_task_image(normalized)


def test_ensure_local_task_image_builds_missing_packaged_image(monkeypatch) -> None:
    normalized = normalize_task_request(
        {
            "task_name": "dugongs",
            "task_path": str(TASK_PACKAGES_DIR / "dugongs"),
            "task_config": {"env": {"name": "dugongs"}},
        }
    )
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def _fake_run(cmd, *args, **kwargs):
        env = kwargs.get("env")
        calls.append((list(cmd), env))
        if cmd[:3] == ["docker", "image", "inspect"]:
            raise RuntimeError("missing image")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("synthstats.integrations.harbor.generator.subprocess.run", _fake_run)

    ensure_local_task_image(normalized)

    assert calls[0][0][:3] == ["docker", "image", "inspect"]
    assert calls[1][0][:3] == ["docker", "build", "-t"]
    assert calls[1][0][3] == "synthstats-harbor-e2e:dugongs"


@pytest.mark.parametrize(
    ("task_name", "instruction_snippet"),
    [
        ("dugongs", "dugong growth curves"),
        ("peregrines", "falcon population dynamics"),
        ("eight_schools", "eight schools"),
        ("surgical", "hospital death counts"),
    ],
)
def test_registered_task_packages_have_local_harbor_files(
    task_name: str,
    instruction_snippet: str,
):
    assert task_name in REGISTERED_TASK_PACKAGES

    package_path = resolve_task_package_path(task_name)
    assert package_path == TASK_PACKAGES_DIR / task_name
    assert (package_path / "task.toml").is_file()
    assert (package_path / "environment" / "Dockerfile").is_file()
    assert (package_path / "tests" / "test.sh").is_file()
    assert instruction_snippet in (package_path / "instruction.md").read_text()


@pytest.mark.parametrize(
    "task_name",
    ["dugongs", "peregrines", "eight_schools", "surgical"],
)
def test_build_local_task_request_uses_real_task_package(task_name: str):
    request = build_local_task_request(
        {
            "env": {
                "name": task_name,
                "_target_": "synthstats.envs.task_runtime_env.TaskRuntimeEnv",
                "task": {"env_name": task_name},
            },
            "judge": {"_target_": "synthstats.judges.likelihood.LikelihoodJudge"},
        }
    )

    assert request["task_name"] == task_name
    assert Path(request["task_path"]) == TASK_PACKAGES_DIR / task_name
    assert request["source"] == "local_task_package"
    assert request["task_config"]["env"]["name"] == task_name
    assert request["task_config"]["judge"]["_target_"].endswith("LikelihoodJudge")


def test_build_local_task_request_omits_empty_top_level_judge_stub():
    request = build_local_task_request(
        {
            "env": {
                "name": "dugongs",
                "_target_": "synthstats.envs.task_runtime_env.TaskRuntimeEnv",
                "task": {"env_name": "dugongs"},
                "judge": {
                    "_target_": "synthstats.judges.likelihood.LikelihoodJudge",
                    "loo_gate_mode": "strict",
                },
            },
            "judge": {
                "_target_": None,
                "judges": [],
                "config": {},
                "type": None,
                "weight": None,
                "scalarization_mode": None,
            },
        }
    )

    assert "judge" not in request["task_config"]
    assert request["task_config"]["env"]["judge"]["_target_"].endswith("LikelihoodJudge")


def test_build_local_task_request_falls_back_to_task_env_name_for_packaged_benchmark():
    request = build_local_task_request(
        {
            "env": {
                "name": "lazyppl_peregrines",
                "_target_": "synthstats.envs.task_runtime_env.TaskRuntimeEnv",
                "task": {"env_name": "peregrines"},
            },
            "judge": {"_target_": "synthstats.judges.likelihood.LikelihoodJudge"},
        }
    )

    assert request["task_name"] == "peregrines"
    assert Path(request["task_path"]) == TASK_PACKAGES_DIR / "peregrines"
    assert request["source"] == "local_task_package"
    assert request["task_config"]["env"]["name"] == "lazyppl_peregrines"
    assert request["task_config"]["env"]["task"]["env_name"] == "peregrines"


def test_unavailable_generator_error_explains_real_install_constraints(monkeypatch):
    monkeypatch.setattr(
        "synthstats.integrations.harbor.generator.platform.system", lambda: "Darwin"
    )
    with pytest.raises(ImportError) as excinfo:
        _unavailable_generator()

    message = str(excinfo.value)
    assert "Linux" in message
    assert "uv sync --extra skyrl --extra ml --extra pymc" in message
    assert "only packaged" not in message


def test_resolve_llm_api_base_prefers_explicit_url():
    assert (
        resolve_llm_api_base(
            "http://relay.example.test:9000/v1",
            inference_port=8000,
            platform_name="Linux",
        )
        == "http://relay.example.test:9000/v1"
    )


def test_resolve_llm_api_base_uses_host_local_default_on_linux():
    assert (
        resolve_llm_api_base(
            None,
            inference_port=8000,
            platform_name="Linux",
        )
        == "http://127.0.0.1:8000/v1"
    )


def test_resolve_llm_api_base_allows_non_linux_default():
    assert (
        resolve_llm_api_base(
            None,
            inference_port=8000,
            platform_name="Darwin",
        )
        == "http://127.0.0.1:8000/v1"
    )


def test_extract_reward_from_trial_result_falls_back_to_verifier_result_without_reward_json():
    trial_result = SimpleNamespace(
        verifier_result=SimpleNamespace(rewards=SimpleNamespace(reward=0.75)),
        agent_result=SimpleNamespace(reward=9.9, metadata={"reward": 9.9}),
    )

    reward, source = extract_reward_from_trial_result(trial_result)

    assert reward == pytest.approx(0.75)
    assert source == "verifier_result"


def test_extract_reward_from_trial_result_prefers_structured_reward_json_over_verifier_result(
    tmp_path,
):
    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier" / "reward.json").write_text(
        '{"schema_version": "synthstats.harbor.reward.v1", '
        '"reward": 0.6, '
        '"reward_source": "validation.elpd", '
        '"loo_status": "accepted", '
        '"components": {"elpd": -3.0}, '
        '"diagnostics": {}}'
    )
    trial_result = SimpleNamespace(
        trial_uri=str(tmp_path),
        verifier_result=SimpleNamespace(rewards=SimpleNamespace(reward=0.75)),
        agent_result=SimpleNamespace(metadata={"reward": 9.9}),
    )

    reward, source = extract_reward_from_trial_result(trial_result)

    assert reward == pytest.approx(0.6)
    assert source == "validation.elpd"


def test_extract_reward_from_trial_result_reads_structured_sidecar(tmp_path):
    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier" / "reward.json").write_text(
        '{"schema_version": "synthstats.harbor.reward.v1", '
        '"reward": 0.6, '
        '"reward_source": "verifier", '
        '"loo_status": "accepted", '
        '"components": {"elpd": -3.0}, '
        '"diagnostics": {}}'
    )
    trial_result = SimpleNamespace(
        trial_uri=str(tmp_path),
        agent_result=SimpleNamespace(metadata={"reward": 9.9}),
    )

    reward, source = extract_reward_from_trial_result(trial_result)

    assert reward == pytest.approx(0.6)
    assert source == "verifier"


def test_extract_reward_signal_from_trial_result_keeps_verifier_semantics(tmp_path):
    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier" / "reward.json").write_text(
        '{"schema_version": "synthstats.harbor.reward.v1", '
        '"reward": 0.6, '
        '"reward_source": "verifier", '
        '"loo_status": "accepted", '
        '"components": {"elpd": -3.0}, '
        '"diagnostics": {}}'
    )
    trial_result = SimpleNamespace(
        trial_uri=str(tmp_path),
        agent_result=SimpleNamespace(metadata={"reward": 9.9}),
    )

    signal = extract_reward_signal_from_trial_result(trial_result)

    assert signal.reward == pytest.approx(0.6)
    assert signal.reward_source == "verifier"
    assert signal.evidence_kind == "verifier"
    assert signal.truth_owner == "verifier"


def test_extract_reward_signal_from_trial_result_prefers_structured_reward_json_over_verifier_result(  # noqa: E501
    tmp_path,
):
    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier" / "reward.json").write_text(
        '{"schema_version": "synthstats.harbor.reward.v1", '
        '"reward": 0.6, '
        '"reward_source": "validation.elpd", '
        '"loo_status": "accepted", '
        '"components": {"elpd": -3.0}, '
        '"diagnostics": {}}'
    )
    trial_result = SimpleNamespace(
        trial_uri=str(tmp_path),
        verifier_result=SimpleNamespace(rewards=SimpleNamespace(reward=0.75)),
        agent_result=SimpleNamespace(metadata={"reward": 9.9}),
    )

    signal = extract_reward_signal_from_trial_result(trial_result)

    assert signal.reward == pytest.approx(0.6)
    assert signal.reward_source == "validation.elpd"
    assert signal.evidence_kind == "verifier"
    assert signal.truth_owner == "verifier"


def test_extract_reward_signal_from_trial_result_uses_agent_reward_info_source():
    trial_result = SimpleNamespace(
        agent_result=SimpleNamespace(
            reward=0.2,
            reward_info={"source": "judge"},
        ),
    )

    signal = extract_reward_signal_from_trial_result(trial_result)

    assert signal.reward == pytest.approx(0.2)
    assert signal.reward_source == "judge"
    assert signal.evidence_kind == "heuristic"
    assert signal.truth_owner == "heuristic"


def test_extract_reward_signal_from_trial_result_treats_validation_sources_as_verifier():
    trial_result = SimpleNamespace(
        agent_result=SimpleNamespace(
            reward=-0.123,
            reward_info={"source": "validation.semantic_score"},
        ),
    )

    signal = extract_reward_signal_from_trial_result(trial_result)

    assert signal.reward == pytest.approx(-0.123)
    assert signal.reward_source == "validation.semantic_score"
    assert signal.evidence_kind == "verifier"
    assert signal.truth_owner == "verifier"


def test_extract_reward_signal_prefers_metadata_reward_info_when_top_level_missing():
    trial_result = SimpleNamespace(
        agent_result=SimpleNamespace(
            reward=0.4,
            metadata={
                "reward": 0.4,
                "reward_info": {"source": "validation.elpd"},
            },
        ),
    )

    signal = extract_reward_signal_from_trial_result(trial_result)

    assert signal.reward == pytest.approx(0.4)
    assert signal.reward_source == "validation.elpd"
    assert signal.evidence_kind == "verifier"
    assert signal.truth_owner == "verifier"


def test_extract_reward_signal_prefers_metadata_reward_info_for_metrics_reward():
    trial_result = SimpleNamespace(
        agent_result=SimpleNamespace(
            metrics={"reward": 0.7},
            metadata={
                "reward_info": {"source": "validation.semantic_score"},
            },
        ),
    )

    signal = extract_reward_signal_from_trial_result(trial_result)

    assert signal.reward == pytest.approx(0.7)
    assert signal.reward_source == "validation.semantic_score"
    assert signal.evidence_kind == "verifier"
    assert signal.truth_owner == "verifier"


def test_extract_reward_signal_supports_dict_shaped_agent_result():
    trial_result = SimpleNamespace(
        agent_result={
            "reward": 0.3,
            "reward_info": {"source": "validation.elpd"},
        },
    )

    signal = extract_reward_signal_from_trial_result(trial_result)

    assert signal.reward == pytest.approx(0.3)
    assert signal.reward_source == "validation.elpd"
    assert signal.evidence_kind == "verifier"
    assert signal.truth_owner == "verifier"


def test_extract_reward_signal_supports_dict_shaped_trial_result_with_verifier():
    trial_result = {
        "verifier_result": {
            "rewards": {
                "reward": 0.8,
            }
        }
    }

    signal = extract_reward_signal_from_trial_result(trial_result)

    assert signal.reward == pytest.approx(0.8)
    assert signal.reward_source == "verifier_result"
    assert signal.evidence_kind == "verifier"
    assert signal.truth_owner == "verifier"


def test_extract_lineage_artifacts_from_atif_reads_dataset_metadata():
    artifacts = extract_lineage_artifacts_from_atif(
        {
            "metadata": {
                "dataset_row": {
                    "id": "completion-001",
                    "env_id": "dugongs",
                    "trace_id": "completion-001:validation",
                },
                "shared_generation_record": {"record_id": "completion-001"},
                "shared_trial_record": {"trial_id": "completion-001"},
                "shared_trace_record": {"trace_id": "completion-001:validation"},
            }
        }
    )

    assert artifacts == {
        "dataset_row": {
            "id": "completion-001",
            "env_id": "dugongs",
            "trace_id": "completion-001:validation",
        },
        "trace_id": "completion-001:validation",
        "task_id": "dugongs",
        "shared_generation_record": {"record_id": "completion-001"},
        "shared_trial_record": {"trial_id": "completion-001"},
        "shared_trace_record": {"trace_id": "completion-001:validation"},
    }


def test_record_rollout_lineage_metrics_counts_lineage_and_verifier_rewards():
    metrics: dict[str, int] = empty_rollout_lineage_metrics()
    reward_signal = extract_reward_signal_from_trial_result(
        SimpleNamespace(verifier_result=SimpleNamespace(rewards=SimpleNamespace(reward=0.9)))
    )

    record_rollout_lineage_metrics(
        metrics,
        artifacts=extract_lineage_artifacts_from_atif(
            {
                "metadata": {
                    "dataset_row": {
                        "id": "completion-001",
                        "env_id": "dugongs",
                        "trace_id": "completion-001:validation",
                    },
                    "shared_generation_record": {"record_id": "completion-001"},
                    "shared_trial_record": {"trial_id": "completion-001"},
                    "shared_trace_record": {"trace_id": "completion-001:validation"},
                }
            }
        ),
        reward_signal=reward_signal,
    )

    assert metrics == {
        "n_with_trace_id": 1,
        "n_with_task_id": 1,
        "n_with_dataset_row": 1,
        "n_with_shared_generation_record": 1,
        "n_with_shared_trial_record": 1,
        "n_with_shared_trace_record": 1,
        "n_with_verifier_reward": 1,
    }


def test_extract_rollout_lineage_flags_emits_rollout_level_binary_signals():
    reward_signal = extract_reward_signal_from_trial_result(
        SimpleNamespace(verifier_result=SimpleNamespace(rewards=SimpleNamespace(reward=0.9)))
    )

    flags = extract_rollout_lineage_flags(
        artifacts=extract_lineage_artifacts_from_atif(
            {
                "metadata": {
                    "dataset_row": {
                        "id": "completion-001",
                        "env_id": "dugongs",
                        "trace_id": "completion-001:validation",
                    },
                    "shared_generation_record": {"record_id": "completion-001"},
                    "shared_trial_record": {"trial_id": "completion-001"},
                    "shared_trace_record": {"trace_id": "completion-001:validation"},
                }
            }
        ),
        reward_signal=reward_signal,
    )

    assert flags == {
        "has_trace_id": 1,
        "has_task_id": 1,
        "has_dataset_row": 1,
        "has_shared_generation_record": 1,
        "has_shared_trial_record": 1,
        "has_shared_trace_record": 1,
        "has_verifier_reward": 1,
    }


def test_record_rollout_lineage_metrics_skips_missing_metadata_and_heuristic_rewards():
    metrics: dict[str, int] = empty_rollout_lineage_metrics()
    reward_signal = extract_reward_signal_from_trial_result(
        SimpleNamespace(
            agent_result=SimpleNamespace(
                reward=0.2,
                reward_info={"source": "judge"},
            )
        )
    )

    record_rollout_lineage_metrics(
        metrics,
        artifacts=extract_lineage_artifacts_from_atif({}),
        reward_signal=reward_signal,
    )

    assert metrics == {
        "n_with_trace_id": 0,
        "n_with_task_id": 0,
        "n_with_dataset_row": 0,
        "n_with_shared_generation_record": 0,
        "n_with_shared_trial_record": 0,
        "n_with_shared_trace_record": 0,
        "n_with_verifier_reward": 0,
    }


def test_empty_rollout_lineage_flags_matches_rollout_level_metric_shape():
    assert empty_rollout_lineage_flags() == {
        "has_trace_id": [],
        "has_task_id": [],
        "has_dataset_row": [],
        "has_shared_generation_record": [],
        "has_shared_trial_record": [],
        "has_shared_trace_record": [],
        "has_verifier_reward": [],
    }


def test_empty_rollout_metrics_matches_live_generator_metric_shape():
    assert empty_rollout_metrics() == {
        "n_success": 0,
        "n_failed": 0,
        "reward_sources": [],
        "reward_evidence_kinds": [],
        "reward_truth_owners": [],
        "has_trace_id": [],
        "has_task_id": [],
        "has_dataset_row": [],
        "has_shared_generation_record": [],
        "has_shared_trial_record": [],
        "has_shared_trace_record": [],
        "has_verifier_reward": [],
        "n_with_trace_id": 0,
        "n_with_task_id": 0,
        "n_with_dataset_row": 0,
        "n_with_shared_generation_record": 0,
        "n_with_shared_trial_record": 0,
        "n_with_shared_trace_record": 0,
        "n_with_verifier_reward": 0,
    }


def test_extract_reward_from_trial_result_falls_back_to_agent_metadata():
    trial_result = SimpleNamespace(
        agent_result=SimpleNamespace(metadata={"reward": 0.2}),
    )

    reward, source = extract_reward_from_trial_result(trial_result)

    assert reward == pytest.approx(0.2)
    assert source == "agent_metadata"


def test_generate_propagates_per_call_sampling_params_with_fallback(monkeypatch):
    import asyncio
    import json
    import sys

    captured_trial_configs = []

    class FakeAgentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.env = kwargs.get("env", {})

    class FakeEnvironmentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTaskConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTrialConfig:
        def __init__(self, **kwargs):
            self.agent = kwargs["agent"]
            self.environment = kwargs["environment"]
            self.task = kwargs["task"]

    class FakeTrial:
        def __init__(self, trial_config):
            captured_trial_configs.append(trial_config)

        @classmethod
        async def create(cls, trial_config):
            return cls(trial_config)

        async def run(self):
            return SimpleNamespace(
                agent_result=SimpleNamespace(
                    reward=1.0,
                    trajectory={"schema_version": "ATIF-v1.5", "steps": []},
                    metadata={
                        "prompt_ids": [1, 2],
                        "token_ids": [3],
                        "mask_ids": [1],
                        "logprobs": [-0.5],
                        "stop_reason": "completed",
                    },
                )
            )

    monkeypatch.setitem(
        sys.modules,
        "harbor.models.trial.config",
        SimpleNamespace(
            AgentConfig=FakeAgentConfig,
            EnvironmentConfig=FakeEnvironmentConfig,
            TaskConfig=FakeTaskConfig,
            TrialConfig=FakeTrialConfig,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "harbor.trial.trial",
        SimpleNamespace(Trial=FakeTrial),
    )
    monkeypatch.setitem(
        sys.modules,
        "harbor.models.trajectories.trajectory",
        SimpleNamespace(Trajectory=SimpleNamespace(model_validate=lambda atif: atif)),
    )

    generator = _build_test_generator(monkeypatch)
    generator.cfg = SimpleNamespace(
        generator=SimpleNamespace(
            max_turns=5,
            sampling_params=SimpleNamespace(
                max_generate_length=10,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
            ),
        ),
        trainer=SimpleNamespace(
            seed=42,
            policy=SimpleNamespace(model=SimpleNamespace(path="Qwen/Qwen3.5-0.8B")),
            algorithm=SimpleNamespace(max_seq_len=2048),
        ),
    )

    eval_batch = {
        "env_extras": [
            {"task_path": "/tmp/external-task", "task_config": {"env": {"name": "ext"}}}
        ],
        "sampling_params": {
            "temperature": 0.0,
            "max_tokens": 1,
            "top_p": 1.0,
            "top_k": -1,
            "stop": ["END"],
        },
    }
    asyncio.run(generator.generate(eval_batch))

    assert len(captured_trial_configs) == 1
    eval_sampling = json.loads(captured_trial_configs[0].agent.env["SAMPLING_PARAMS"])
    assert eval_sampling["temperature"] == pytest.approx(0.0)
    assert eval_sampling["max_generate_length"] == 1
    assert eval_sampling["top_p"] == pytest.approx(1.0)
    assert eval_sampling["top_k"] == -1
    assert eval_sampling["stop"] == ["END"]

    train_batch = {
        "env_extras": [
            {"task_path": "/tmp/external-task", "task_config": {"env": {"name": "ext"}}}
        ],
    }
    asyncio.run(generator.generate(train_batch))

    assert len(captured_trial_configs) == 2
    train_sampling = json.loads(captured_trial_configs[1].agent.env["SAMPLING_PARAMS"])
    assert train_sampling["temperature"] == pytest.approx(0.7)
    assert train_sampling["max_generate_length"] == 10
    assert train_sampling["top_p"] == pytest.approx(0.95)
    assert train_sampling["top_k"] == 50


def test_resolve_local_repo_mount_config_for_installed_wheel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import synthstats.integrations.harbor.generator as generator_mod

    site_packages = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    installed_pkg = site_packages / "synthstats"
    installed_harbor = installed_pkg / "integrations" / "harbor"
    installed_task = installed_harbor / "task_packages" / "dugongs"
    installed_task.mkdir(parents=True)
    (installed_pkg / "__init__.py").write_text("")
    installed_generator = installed_harbor / "generator.py"
    installed_generator.write_text("# installed generator stub\n")

    monkeypatch.setattr(generator_mod, "__file__", str(installed_generator))
    monkeypatch.setattr(generator_mod, "_LOCAL_REPO_ROOT", installed_generator.resolve().parents[4])

    mounts, env = resolve_local_repo_mount_config({"path": str(installed_task)})

    assert mounts is not None
    assert mounts[0]["source"] == str(site_packages.resolve())
    assert mounts[0]["target"] == "/workspace/synthstats"
    assert env == {"PYTHONPATH": "/workspace/synthstats"}
    rel_pythonpath = Path(env["PYTHONPATH"]).relative_to(mounts[0]["target"])
    assert (Path(mounts[0]["source"]) / rel_pythonpath / "synthstats").is_dir()
