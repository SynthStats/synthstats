"""Harbor file URIs and the agent/verifier artifact layout."""

import json

import pytest

from synthstats.integrations.harbor.collector import HarborCollector
from synthstats.integrations.harbor.generator import extract_reward_signal_from_trial_result


def test_file_uri_preserves_structured_verifier_reward(tmp_path):
    trial = tmp_path / "trial with spaces"
    verifier = trial / "verifier"
    verifier.mkdir(parents=True)
    (verifier / "reward.json").write_text(
        json.dumps(
            {
                "reward": 0.7,
                "reward_source": "validation.likelihood",
                "evidence_kind": "verifier",
                "truth_owner": "verifier",
            }
        )
    )
    signal = extract_reward_signal_from_trial_result(
        {"trial_uri": trial.as_uri(), "verifier_result": {"rewards": {"reward": 0.7}}}
    )
    assert signal.reward == pytest.approx(0.7)
    assert signal.reward_source == "validation.likelihood"
    assert signal.evidence_kind == "verifier"


def test_collector_reads_verifier_sibling_of_agent_directory(tmp_path):
    agent = tmp_path / "trial" / "agent"
    verifier = agent.parent / "verifier"
    agent.mkdir(parents=True)
    verifier.mkdir()
    (agent / "trajectory.json").write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.6",
                "steps": [],
            }
        )
    )
    (verifier / "reward.json").write_text(
        json.dumps(
            {
                "reward": 0.7,
                "reward_source": "validation.likelihood",
                "evidence_kind": "verifier",
                "truth_owner": "verifier",
            }
        )
    )
    collected = HarborCollector(tmp_path, shuffle=False, cycle=False).collect_atif()
    assert len(collected) == 1
    assert collected[0][1] == pytest.approx(0.7)


def test_unsupported_trial_uri_is_rejected():
    with pytest.raises(ValueError, match="scheme"):
        extract_reward_signal_from_trial_result({"trial_uri": "https://example.invalid/trial"})
