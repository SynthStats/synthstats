from __future__ import annotations

import json

from synthstats.core.types import Reward
from synthstats.integrations.harbor.verifier import run_verifier, verify_agent_result


def test_verifier_replay_rejects_printed_score_without_model():
    artifact = verify_agent_result(
        {
            "program": "print('__ELPD__:2.0')",
            "language": "pymc",
            "reward": 7.38905609893065,
            "task_config": {
                "env": {
                    "_target_": "synthstats.envs.task_runtime_env.TaskRuntimeEnv",
                    "task": {
                        "_target_": (
                            "synthstats.tasks.statistical_modeling.task.StatisticalModelingTask"
                        ),
                        "env_name": "dugongs",
                    },
                    "codec": {"_target_": "synthstats.runtime.codecs.NativeToolCallingCodec"},
                },
                "judge": {
                    "_target_": "synthstats.judges.likelihood.LikelihoodJudge",
                    "beta": 1.0,
                },
            },
        }
    )
    assert artifact["reward"] < 1.0
    assert artifact["elpd"] != 2.0
    assert artifact["loo_status"] != "accepted"


def test_verify_agent_result_builds_canonical_reward_artifact(monkeypatch) -> None:
    class _Env:
        def score_program_reward(self, program: str, language: str = "pymc") -> Reward:
            assert program == "x = 1"
            assert language == "pymc"
            return Reward(
                total=0.75,
                components={"elpd": -3.0},
                info={
                    "reward_source": "validation.likelihood",
                    "loo_status": "accepted",
                    "diagnostics": {"warning": False},
                    "would_reject_strict": False,
                },
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "synthstats.envs.builders.build_env_from_dict",
        lambda cfg: _Env(),
    )

    artifact = verify_agent_result(
        {
            "program": "x = 1",
            "language": "pymc",
            "reward": 0.5,
            "task_config": {
                "env": {
                    "name": "dugongs",
                    "_target_": "synthstats.envs.task_runtime_env.TaskRuntimeEnv",
                },
                "judge": {
                    "_target_": "synthstats.judges.likelihood.LikelihoodJudge",
                },
            },
            "task_provenance": {
                "source": "local_task_package",
                "path": "/tmp/dugongs",
            },
        }
    )

    assert artifact["schema_version"] == "synthstats.harbor.reward.v1"
    assert artifact["reward"] == 0.75
    assert artifact["reward_source"] == "validation.likelihood"
    assert artifact["evidence_kind"] == "verifier"
    assert artifact["truth_owner"] == "verifier"
    assert artifact["elpd"] == -3.0
    assert artifact["loo_status"] == "accepted"
    assert artifact["diagnostics"] == {"warning": False}
    assert artifact["provenance"]["task_name"] == "dugongs"
    assert artifact["provenance"]["language"] == "pymc"
    assert artifact["provenance"]["env_target"].endswith("TaskRuntimeEnv")
    assert artifact["provenance"]["judge_target"].endswith("LikelihoodJudge")
    assert artifact["provenance"]["task_ref"] == {
        "source": "local_task_package",
        "path": "/tmp/dugongs",
    }


def test_run_verifier_writes_reward_json_and_txt(monkeypatch, tmp_path) -> None:
    class _Env:
        def score_program_reward(self, program: str, language: str = "pymc") -> Reward:
            return Reward(
                total=0.6,
                components={"elpd": -2.0},
                info={
                    "reward_source": "validation.elpd",
                    "loo_status": "accepted",
                    "diagnostics": {},
                },
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "synthstats.envs.builders.build_env_from_dict",
        lambda cfg: _Env(),
    )

    agent_result_path = tmp_path / "logs" / "agent" / "synthstats_result.json"
    agent_result_path.parent.mkdir(parents=True, exist_ok=True)
    agent_result_path.write_text(
        json.dumps(
            {
                "program": "x = 1",
                "language": "pymc",
                "task_config": {"env": {"name": "dugongs"}},
            }
        )
    )

    artifact = run_verifier(
        agent_result_path=agent_result_path,
        output_dir=tmp_path / "logs" / "verifier",
    )

    reward_json = json.loads((tmp_path / "logs" / "verifier" / "reward.json").read_text())
    reward_artifact = json.loads(
        (tmp_path / "logs" / "verifier" / "reward_artifact.json").read_text()
    )
    reward_txt = (tmp_path / "logs" / "verifier" / "reward.txt").read_text().strip()

    assert artifact["reward"] == 0.6
    assert artifact["reward_source"] == "validation.elpd"
    assert reward_json["reward"] == 0.6
    assert reward_json == {"reward": 0.6}
    assert reward_artifact == artifact
    assert reward_txt == "0.6"


def test_verify_agent_result_preserves_heuristic_failure_semantics(monkeypatch) -> None:
    class _Env:
        def score_program_reward(self, program: str, language: str = "pymc") -> Reward:
            return Reward(
                total=1e-8,
                components={},
                info={
                    "source": "judge_error",
                    "judge_error": "backend exploded",
                    "reward_floor_applied": True,
                },
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "synthstats.envs.builders.build_env_from_dict",
        lambda cfg: _Env(),
    )

    artifact = verify_agent_result(
        {
            "program": "x = 1",
            "language": "pymc",
            "task_config": {"env": {"name": "dugongs"}},
        }
    )

    assert artifact["reward_source"] == "judge_error"
    assert artifact["evidence_kind"] == "heuristic"
    assert artifact["truth_owner"] == "heuristic"
    assert artifact["info"]["judge_error"] == "backend exploded"
