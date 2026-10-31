"""The Harbor wire reward and the detailed verifier record round-trip together."""

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from synthstats.core.types import Reward
from synthstats.integrations.harbor.collector import HarborCollector
from synthstats.integrations.harbor.converter import read_reward_sidecar
from synthstats.integrations.harbor.generator import extract_reward_signal_from_trial_result
from synthstats.integrations.harbor.reward_artifacts import (
    build_reward_artifact,
    write_reward_artifacts,
)


def test_writer_emits_numeric_harbor_reward_and_preserves_verifier_details(tmp_path):
    artifact = build_reward_artifact(
        Reward(
            total=0.6,
            components={"elpd": -2.0},
            info={"loo_status": "accepted", "diagnostics": {"warning": False}},
        ),
        reward_source="validation.likelihood",
        provenance={"task_name": "dugongs"},
    )
    reward_json, reward_txt = write_reward_artifacts(artifact, output_dir=tmp_path)

    assert json.loads(reward_json.read_text()) == {"reward": 0.6}
    assert reward_txt.read_text() == "0.6\n"
    assert json.loads((tmp_path / "reward_artifact.json").read_text()) == artifact


@pytest.mark.parametrize("name", ["reward.json", "reward.txt", "reward_artifact.json"])
def test_sidecar_reader_preserves_companion_semantics(name, tmp_path):
    (tmp_path / "reward.json").write_text('{"reward": 0.6}')
    (tmp_path / "reward.txt").write_text("0.6\n")
    (tmp_path / "reward_artifact.json").write_text(
        json.dumps(
            {
                "reward": 0.6,
                "reward_source": "validation.likelihood",
                "evidence_kind": "verifier",
                "truth_owner": "verifier",
                "components": {"elpd": -2.0},
                "loo_status": "accepted",
                "diagnostics": {"warning": False},
                "provenance": {"task_name": "dugongs"},
            }
        )
    )
    reward = read_reward_sidecar(str(tmp_path / name))
    assert reward is not None
    assert reward.total == 0.6
    assert reward.components == {"elpd": -2.0}
    assert reward.info["reward_source"] == "validation.likelihood"
    assert reward.info["evidence_kind"] == "verifier"
    assert reward.info["truth_owner"] == "verifier"
    assert reward.info["loo_status"] == "accepted"
    assert reward.info["diagnostics"] == {"warning": False}
    assert reward.info["provenance"] == {"task_name": "dugongs"}


def test_generator_and_collector_read_rich_companion_without_wire_files(tmp_path):
    agent = tmp_path / "trial" / "agent"
    verifier = agent.parent / "verifier"
    agent.mkdir(parents=True)
    verifier.mkdir()
    (agent / "trajectory.json").write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.6",
                "steps": [],
                "final_metrics": {"extra": {"reward": 9.0}},
            }
        )
    )
    (verifier / "reward_artifact.json").write_text(
        json.dumps(
            {
                "reward": 0.6,
                "reward_source": "validation.likelihood",
                "evidence_kind": "verifier",
                "truth_owner": "verifier",
            }
        )
    )
    signal = extract_reward_signal_from_trial_result(
        {
            "trial_uri": agent.parent.as_uri(),
            "verifier_result": {"rewards": {"reward": 9.0}},
        }
    )
    assert signal.reward == 0.6
    assert signal.reward_source == "validation.likelihood"
    assert signal.truth_owner == "verifier"
    collector = HarborCollector(tmp_path, shuffle=False, cycle=False)
    collected = collector.collect_atif()
    assert len(collected) == 1
    assert collected[0][1] == 0.6
    assert collector.last_collection_metrics["episodes_with_verifier_reward"] == 1


def test_written_companion_round_trips_through_consumers(tmp_path):
    agent = tmp_path / "trial" / "agent"
    verifier = agent.parent / "verifier"
    agent.mkdir(parents=True)
    (agent / "trajectory.json").write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.6",
                "steps": [
                    {
                        "step_id": 1,
                        "source": "agent",
                        "message": "model",
                        "metrics": {"completion_token_ids": [10], "logprobs": [-0.1]},
                    }
                ],
            }
        )
    )
    artifact = build_reward_artifact(
        Reward(
            total=0.6,
            components={"elpd": -2.0},
            info={"loo_status": "accepted", "diagnostics": {"warning": False}},
        ),
        reward_source="validation.likelihood",
        provenance={"task_name": "dugongs"},
    )
    write_reward_artifacts(artifact, output_dir=verifier)
    recovered = read_reward_sidecar(str(verifier / "reward.json"))
    assert recovered is not None
    assert recovered.total == 0.6
    assert recovered.components == {"elpd": -2.0}
    assert recovered.info["elpd"] == -2.0
    assert recovered.info["loo_status"] == "accepted"
    assert recovered.info["diagnostics"] == {"warning": False}
    assert recovered.info["provenance"] == {"task_name": "dugongs"}
    assert recovered.info["reward_source"] == "validation.likelihood"
    assert recovered.info["evidence_kind"] == "verifier"
    assert recovered.info["truth_owner"] == "verifier"
    signal = extract_reward_signal_from_trial_result(
        {
            "trial_uri": agent.parent.as_uri(),
            "verifier_result": {"rewards": {"reward": 9.0}},
        }
    )
    assert signal.reward == 0.6
    assert signal.reward_source == "validation.likelihood"
    assert signal.evidence_kind == "verifier"
    assert signal.truth_owner == "verifier"
    collector = HarborCollector(tmp_path, shuffle=False, cycle=False)
    collected = collector.collect()
    assert len(collected) == 1
    assert collected[0].reward == 0.6
    assert collector.last_collection_metrics["episodes_with_verifier_reward"] == 1


@pytest.mark.parametrize("consumer", ["sidecar", "generator", "collector"])
@pytest.mark.parametrize("payload", ["{", "[]", "{}", '{"reward": "bad"}', '{"reward": 0.6}'])
def test_corrupt_companion_cannot_fall_back_to_numeric_reward(consumer, payload, tmp_path):
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "reward.json").write_text('{"reward": 0.6}')
    (verifier / "reward_artifact.json").write_text(payload)
    with pytest.raises(ValueError):
        if consumer == "sidecar":
            read_reward_sidecar(str(verifier / "reward.json"))
        elif consumer == "generator":
            extract_reward_signal_from_trial_result(
                {"trial_uri": tmp_path.as_uri(), "verifier_result": {"rewards": {"reward": 0.6}}}
            )
        else:
            agent = tmp_path / "agent"
            agent.mkdir()
            (agent / "trajectory.json").write_text(
                json.dumps({"schema_version": "ATIF-v1.6", "steps": []})
            )
            HarborCollector(tmp_path, shuffle=False, cycle=False).collect_atif()


@pytest.mark.parametrize("consumer", ["sidecar", "generator", "collector"])
def test_synthstats_trial_requires_its_rich_reward_companion(consumer, tmp_path):
    verifier = tmp_path / "verifier"
    agent = tmp_path / "agent"
    verifier.mkdir()
    agent.mkdir()
    (agent / "synthstats_result.json").write_text('{"language": "pymc"}')
    (agent / "trajectory.json").write_text(json.dumps({"schema_version": "ATIF-v1.6", "steps": []}))
    (verifier / "reward.json").write_text('{"reward": 0.6}')
    with pytest.raises(FileNotFoundError, match="reward_artifact.json"):
        if consumer == "sidecar":
            read_reward_sidecar(str(verifier / "reward.json"))
        elif consumer == "generator":
            extract_reward_signal_from_trial_result(
                {"trial_uri": tmp_path.as_uri(), "verifier_result": {"rewards": {"reward": 0.6}}}
            )
        else:
            HarborCollector(tmp_path, shuffle=False, cycle=False).collect_atif()


@pytest.mark.parametrize(
    "field,value",
    [
        ("reward", None),
        ("reward", True),
        ("reward", float("nan")),
        ("reward", float("inf")),
        ("reward", "0.6"),
        ("reward_source", None),
        ("evidence_kind", ""),
        ("truth_owner", " "),
    ],
)
def test_companion_rejects_invalid_reward_and_ownership_fields(field, value, tmp_path):
    payload = {
        "reward": 0.6,
        "reward_source": "validation.likelihood",
        "evidence_kind": "verifier",
        "truth_owner": "verifier",
    }
    payload[field] = value
    path = tmp_path / "reward_artifact.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        read_reward_sidecar(str(path))


@given(st.floats(min_value=1e-8, max_value=1e8, allow_nan=False, allow_infinity=False))
def test_reward_details_round_trip_for_finite_rewards(total):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        artifact = build_reward_artifact(
            Reward(total=total, components={"elpd": -2.0}, info={"loo_status": "accepted"}),
            reward_source="validation.likelihood",
            provenance={"task_name": "dugongs"},
        )
        write_reward_artifacts(artifact, output_dir=directory)
        recovered = read_reward_sidecar(str(Path(directory) / "reward_artifact.json"))
        assert recovered is not None
        assert recovered.total == total
        assert recovered.components == {"elpd": -2.0}
        assert recovered.info["reward_source"] == "validation.likelihood"
        assert recovered.info["evidence_kind"] == "verifier"
        assert recovered.info["truth_owner"] == "verifier"
        assert recovered.info["loo_status"] == "accepted"
        assert recovered.info["provenance"] == {"task_name": "dugongs"}
