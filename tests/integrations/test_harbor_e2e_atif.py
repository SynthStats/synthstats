"""End-to-end Harbor collector and converter tests on ATIF v1.6 trajectory directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _make_atif_record(
    session_id: str,
    *,
    program: str,
    reward: float,
    reward_source: str,
    evidence_kind: str,
    truth_owner: str,
    verification_stale: bool | None = None,
) -> dict[str, Any]:
    final_metrics: dict[str, Any] = {
        "reward": reward,
        "reward_source": reward_source,
        "evidence_kind": evidence_kind,
        "truth_owner": truth_owner,
        "source": "final_metrics",
    }
    if verification_stale is not None:
        final_metrics["verification_stale"] = verification_stale
    return {
        "schema_version": "ATIF-v1.6",
        "session_id": session_id,
        "agent": {"name": "synthstats-agent", "version": "0.1.0", "model_name": "qwen"},
        "steps": [
            {"step_id": 1, "source": "system", "message": "System instruction."},
            {"step_id": 2, "source": "user", "message": "Fit the model."},
            {
                "step_id": 3,
                "source": "agent",
                "message": "Submitting program.",
                "tool_calls": [
                    {
                        "tool_call_id": "call-1",
                        "function_name": "submit_program",
                        "arguments": {"code": program, "language": "haskell"},
                    }
                ],
                "metrics": {
                    "prompt_tokens": 12,
                    "completion_tokens": 4,
                    "prompt_token_ids": [1, 2, 3],
                    "completion_token_ids": [10, 11, 12, 13],
                    "logprobs": [-0.2, -0.3, -0.1, -0.4],
                },
            },
        ],
        "final_metrics": {"extra": final_metrics},
    }


@pytest.fixture
def atif_dir(tmp_path: Path) -> Path:
    records = {
        "case-dugongs-0000.json": _make_atif_record(
            "case-dugongs-0000",
            program="module Submission where\nmodel :: Int\nmodel = 1\n",
            reward=-12.5,
            reward_source="validation.elpd",
            evidence_kind="verifier",
            truth_owner="verifier",
        ),
        "case-dugongs-0001.json": _make_atif_record(
            "case-dugongs-0001",
            program="module Submission where\nmodel :: Int\nmodel = 2\n",
            reward=0.25,
            reward_source="heuristic.code_change",
            evidence_kind="heuristic",
            truth_owner="heuristic",
            verification_stale=True,
        ),
        "sft-lazyppl-0000.json": _make_atif_record(
            "sft-lazyppl-0000",
            program="module Submission where\nmodel :: Int\nmodel = 3\n",
            reward=0.5,
            reward_source="heuristic.runtime_import",
            evidence_kind="heuristic",
            truth_owner="heuristic",
        ),
    }
    for filename, payload in records.items():
        (tmp_path / filename).write_text(json.dumps(payload), encoding="utf-8")

    (tmp_path / "case-dugongs-0001.json.bak").write_text("{}", encoding="utf-8")
    manifest = {
        "count": len(records),
        "files": [
            {
                "file": name,
                "reward_source": payload["final_metrics"]["extra"]["reward_source"],
                "evidence_kind": payload["final_metrics"]["extra"]["evidence_kind"],
                "truth_owner": payload["final_metrics"]["extra"]["truth_owner"],
            }
            for name, payload in sorted(records.items())
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


class TestHarborE2EATIF:
    def test_collector_scans_correct_file_count(self, atif_dir: Path):
        from synthstats.integrations.harbor.collector import HarborCollector

        c = HarborCollector(atif_dir, shuffle=False)
        assert c.num_files == 3

    def test_collector_loads_atif_records(self, atif_dir: Path):
        from synthstats.integrations.harbor.collector import HarborCollector

        c = HarborCollector(atif_dir, shuffle=False)
        pairs = c.collect_atif(episodes=3)
        assert len(pairs) == 3
        for atif, reward in pairs:
            assert isinstance(atif, dict)
            assert "schema_version" in atif
            assert atif["schema_version"].startswith("ATIF-v1.")
            assert "steps" in atif
            assert isinstance(reward, float)

    def test_collected_atif_schema_is_v16(self, atif_dir: Path):
        from synthstats.integrations.harbor.collector import HarborCollector

        c = HarborCollector(atif_dir, shuffle=False)
        pairs = c.collect_atif(episodes=3)
        for atif, _ in pairs:
            assert atif["schema_version"] == "ATIF-v1.6", (
                f"Expected ATIF-v1.6, got {atif['schema_version']!r} "
                f"in session {atif.get('session_id')}"
            )

    def test_reward_comes_from_final_metrics(self, atif_dir: Path):
        from synthstats.integrations.harbor.collector import HarborCollector

        c = HarborCollector(atif_dir, shuffle=False)
        pairs = c.collect_atif(episodes=3)
        for atif, reward in pairs:
            final = atif["final_metrics"]["extra"]
            assert "reward" in final
            assert reward == pytest.approx(float(final["reward"]), rel=1e-6)

    def test_collected_files_store_explicit_verifier_semantics(self, atif_dir: Path):
        sample_file = sorted(atif_dir.glob("case-dugongs-*.json"))[0]
        atif = json.loads(sample_file.read_text())
        final_metrics = atif["final_metrics"]["extra"]

        assert final_metrics["reward_source"] == "validation.elpd"
        assert final_metrics["evidence_kind"] == "verifier"
        assert final_metrics["truth_owner"] == "verifier"
        assert final_metrics["source"] == "final_metrics"

    def test_changed_programs_mark_unrevalidated_rewards_as_heuristic(self, atif_dir: Path):
        atif = json.loads((atif_dir / "case-dugongs-0001.json").read_text())
        final_metrics = atif["final_metrics"]["extra"]

        assert final_metrics["reward_source"] == "heuristic.code_change"
        assert final_metrics["evidence_kind"] == "heuristic"
        assert final_metrics["truth_owner"] == "heuristic"
        assert final_metrics["verification_stale"] is True

    def test_sft_files_store_explicit_heuristic_semantics(self, atif_dir: Path):
        sample_file = sorted(atif_dir.glob("sft-lazyppl-*.json"))[0]
        atif = json.loads(sample_file.read_text())
        final_metrics = atif["final_metrics"]["extra"]

        assert final_metrics["reward_source"] == "heuristic.runtime_import"
        assert final_metrics["evidence_kind"] == "heuristic"
        assert final_metrics["truth_owner"] == "heuristic"
        assert final_metrics["source"] == "final_metrics"

    def test_manifest_entries_store_explicit_reward_provenance(self, atif_dir: Path):
        manifest = json.loads((atif_dir / "manifest.json").read_text())

        assert manifest["count"] == len(manifest["files"])
        sft_entry = next(entry for entry in manifest["files"] if entry["file"].startswith("sft-"))
        assert sft_entry["reward_source"] == "heuristic.runtime_import"
        assert sft_entry["evidence_kind"] == "heuristic"
        assert sft_entry["truth_owner"] == "heuristic"

    def test_all_files_have_token_data(self, atif_dir: Path):
        from synthstats.integrations.harbor.collector import HarborCollector

        c = HarborCollector(atif_dir, shuffle=False)
        pairs = c.collect_atif(episodes=c.num_files)
        assert len(pairs) == c.num_files
        for atif, _ in pairs:
            has_tokens = any(
                step.get("source") == "agent"
                and step.get("metrics", {}).get("completion_token_ids")
                for step in atif.get("steps", [])
            )
            assert has_tokens, f"session {atif.get('session_id')} has no agent step with token data"

    def test_atif_to_trajectory_on_collected_file(self, atif_dir: Path):
        from synthstats.core.types import Reward
        from synthstats.integrations.harbor.converter import atif_to_trajectory

        sample_file = sorted(atif_dir.glob("case-dugongs-*.json"))[0]
        atif = json.loads(sample_file.read_text())
        reward = Reward(total=1e-8, components={}, info={})

        traj = atif_to_trajectory(atif, reward)

        assert len(traj.messages) >= 3  # system + user + assistant
        assert len(traj.token_ids) == 1  # one agent generation
        assert len(traj.token_logprobs) == 1
        assert len(traj.token_ids[0]) > 0
        assert len(traj.token_logprobs[0]) == len(traj.token_ids[0])
        assert traj.reward.total == pytest.approx(1e-8)

    def test_extract_program_from_collected_file(self, atif_dir: Path):
        from synthstats.integrations.harbor.converter import extract_program

        for fname in sorted(atif_dir.glob("case-dugongs-*.json")):
            atif = json.loads(fname.read_text())
            prog = extract_program(atif)
            if prog is not None:
                assert "module Submission" in prog or len(prog) > 10
                break
        else:
            pytest.fail("no dugongs ATIF files with programs found")

    def test_bak_files_excluded_from_scan(self, atif_dir: Path):
        from synthstats.integrations.harbor.collector import HarborCollector

        c = HarborCollector(atif_dir, shuffle=False)
        for path in c._files:
            assert not path.name.endswith(".bak"), (
                f"_scan_files should not include .bak files: {path}"
            )

    def test_collector_cycles_over_dataset(self, atif_dir: Path):
        from synthstats.integrations.harbor.collector import HarborCollector

        c = HarborCollector(atif_dir, shuffle=False, cycle=True)
        n = c.num_files * 2
        pairs = c.collect_atif(episodes=n)
        assert len(pairs) == n
        assert c.epoch == 1
