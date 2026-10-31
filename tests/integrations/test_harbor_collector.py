"""Tests for HarborCollector (reads ATIF files from disk)."""

from __future__ import annotations

import json

import pytest
import torch

from synthstats.core.types import Reward
from synthstats.integrations.harbor.collector import HarborCollector
from synthstats.train.data.collectors import CollectedTrajectory


def _write_atif(path, *, reward_total: float = 0.7):
    """Write a minimal valid ATIF file + plain sidecar reward.json."""
    atif = {
        "schema_version": "ATIF-v1.5",
        "final_metrics": {"extra": {"reward": reward_total / 10.0}},
        "steps": [
            {"step_id": 1, "source": "system", "message": "system prompt"},
            {"step_id": 2, "source": "user", "message": "write a model"},
            {
                "step_id": 3,
                "source": "agent",
                "message": "here is the model",
                "metrics": {
                    "completion_token_ids": [10, 20, 30],
                    "logprobs": [-0.1, -0.2, -0.3],
                },
            },
        ],
    }
    path.write_text(json.dumps(atif))

    reward_path = path.parent / "reward.json"
    if not reward_path.exists():
        reward_path.write_text(json.dumps({"reward": reward_total}))


@pytest.fixture()
def atif_dir(tmp_path):
    """Create a temp directory with 5 ATIF trajectory files."""
    for i in range(5):
        task_dir = tmp_path / f"task_{i}"
        task_dir.mkdir()
        _write_atif(task_dir / "trajectory.json", reward_total=0.1 * (i + 1))
    return tmp_path


class TestHarborCollector:
    def test_init_with_valid_dir(self, atif_dir):
        c = HarborCollector(atif_dir, shuffle=False)
        assert c.num_files == 5

    def test_init_missing_dir(self):
        with pytest.raises(FileNotFoundError):
            HarborCollector("/nonexistent/dir")

    def test_init_empty_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no ATIF JSON"):
            HarborCollector(tmp_path)

    def test_collect_returns_collected_trajectory(self, atif_dir):
        c = HarborCollector(atif_dir, shuffle=False)
        results = c.collect(episodes=2)
        assert len(results) == 2
        for ct in results:
            assert isinstance(ct, CollectedTrajectory)
            assert isinstance(ct.log_probs, torch.Tensor)
            assert isinstance(ct.entropy, torch.Tensor)
            assert isinstance(ct.reward, float)

    def test_collect_cycles(self, atif_dir):
        c = HarborCollector(atif_dir, shuffle=False, cycle=True)
        # collect more than available files
        results = c.collect(episodes=7)
        assert len(results) == 7
        assert c.epoch >= 1

    def test_collect_no_cycle_exhausts(self, atif_dir):
        c = HarborCollector(atif_dir, shuffle=False, cycle=False)
        results = c.collect(episodes=10)
        assert len(results) == 5

    def test_collect_reads_sidecar_reward(self, atif_dir):
        c = HarborCollector(atif_dir, shuffle=False)
        results = c.collect(episodes=1)
        assert len(results) == 1
        assert results[0].reward > 0.0
        assert results[0].reward == pytest.approx(0.1)

    def test_collector_protocol_conformance(self, atif_dir):
        """HarborCollector should match the Collector protocol signature."""
        c = HarborCollector(atif_dir, shuffle=False)
        # protocol requires: collect(episodes, *, temperature, compute_ref_log_probs)
        results = c.collect(episodes=1, temperature=0.8, compute_ref_log_probs=False)
        assert len(results) == 1

    def test_collect_with_observations_and_actions(self, atif_dir):
        c = HarborCollector(atif_dir, shuffle=False)
        results = c.collect(episodes=1)
        ct = results[0]
        assert len(ct.observations) > 0
        assert len(ct.actions) > 0

    def test_collect_preserves_atif_lineage_artifacts(self, tmp_path):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        atif = {
            "schema_version": "ATIF-v1.6",
            "session_id": "sft-lazyppl-0007",
            "metadata": {
                "source": "synthstats.dataset_row",
                "dataset_row": {
                    "id": "completion-001",
                    "env_id": "dugongs",
                    "trace_id": "completion-001:validation",
                },
                "shared_trace_record": {
                    "trace_id": "completion-001:validation",
                    "trial_id": "completion-001",
                },
            },
            "final_metrics": {"extra": {"reward": 0.42}},
            "steps": [
                {"step_id": 1, "source": "system", "message": "system prompt"},
                {"step_id": 2, "source": "user", "message": "write a model"},
                {
                    "step_id": 3,
                    "source": "agent",
                    "message": "here is the model",
                    "metrics": {
                        "completion_token_ids": [10, 20, 30],
                        "logprobs": [-0.1, -0.2, -0.3],
                    },
                },
            ],
        }
        (task_dir / "trajectory.json").write_text(json.dumps(atif))

        c = HarborCollector(tmp_path, shuffle=False)
        results = c.collect(episodes=1)

        assert len(results) == 1
        assert results[0].artifacts["session_id"] == "sft-lazyppl-0007"
        assert results[0].artifacts["task_id"] == "dugongs"
        assert results[0].artifacts["trace_id"] == "completion-001:validation"
        assert results[0].artifacts["shared_trace_record"]["trial_id"] == "completion-001"

    def test_collect_records_lineage_monitoring_metrics(self, tmp_path):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        atif = {
            "schema_version": "ATIF-v1.6",
            "session_id": "sft-lazyppl-0007",
            "metadata": {
                "source": "synthstats.dataset_row",
                "dataset_row": {
                    "id": "completion-001",
                    "env_id": "dugongs",
                    "trace_id": "completion-001:validation",
                },
                "shared_generation_record": {"record_id": "completion-001"},
                "shared_trial_record": {"trial_id": "completion-001"},
                "shared_trace_record": {"trace_id": "completion-001:validation"},
            },
            "final_metrics": {"extra": {"reward": 0.42}},
            "steps": [
                {"step_id": 1, "source": "system", "message": "system prompt"},
                {"step_id": 2, "source": "user", "message": "write a model"},
                {
                    "step_id": 3,
                    "source": "agent",
                    "message": "here is the model",
                    "metrics": {
                        "completion_token_ids": [10, 20, 30],
                        "logprobs": [-0.1, -0.2, -0.3],
                    },
                },
            ],
        }
        (task_dir / "trajectory.json").write_text(json.dumps(atif))
        (task_dir / "reward.json").write_text(
            json.dumps(
                {
                    "schema_version": "synthstats.harbor.reward.v1",
                    "reward": 0.9,
                    "reward_source": "verifier",
                    "evidence_kind": "verifier",
                    "truth_owner": "verifier",
                    "components": {"elpd": -1.0},
                    "loo_status": "accepted",
                    "diagnostics": {},
                    "info": {},
                }
            )
        )

        c = HarborCollector(tmp_path, shuffle=False)
        c.collect(episodes=1)

        assert c.last_collection_metrics == {
            "episodes_collected": 1,
            "episodes_with_trace_id": 1,
            "episodes_with_task_id": 1,
            "episodes_with_dataset_row": 1,
            "episodes_with_shared_generation_record": 1,
            "episodes_with_shared_trial_record": 1,
            "episodes_with_shared_trace_record": 1,
            "episodes_with_verifier_reward": 1,
        }

    def test_collect_atif_records_lineage_monitoring_metrics(self, tmp_path):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        atif = {
            "schema_version": "ATIF-v1.6",
            "session_id": "sft-lazyppl-0007",
            "metadata": {
                "source": "synthstats.dataset_row",
                "dataset_row": {
                    "id": "completion-001",
                    "env_id": "dugongs",
                    "trace_id": "completion-001:validation",
                },
                "shared_trace_record": {"trace_id": "completion-001:validation"},
            },
            "final_metrics": {"extra": {"reward": 0.42}},
            "steps": [
                {"step_id": 1, "source": "system", "message": "system prompt"},
                {"step_id": 2, "source": "user", "message": "write a model"},
                {
                    "step_id": 3,
                    "source": "agent",
                    "message": "here is the model",
                    "metrics": {
                        "completion_token_ids": [10, 20, 30],
                        "logprobs": [-0.1, -0.2, -0.3],
                    },
                },
            ],
        }
        (task_dir / "trajectory.json").write_text(json.dumps(atif))
        (task_dir / "reward.json").write_text(
            json.dumps(
                {
                    "schema_version": "synthstats.harbor.reward.v1",
                    "reward": 0.9,
                    "reward_source": "verifier",
                    "evidence_kind": "verifier",
                    "truth_owner": "verifier",
                    "components": {"elpd": -1.0},
                    "loo_status": "accepted",
                    "diagnostics": {},
                    "info": {},
                }
            )
        )

        c = HarborCollector(tmp_path, shuffle=False)
        rows = c.collect_atif(episodes=1)

        assert len(rows) == 1
        assert c.last_collection_metrics == {
            "episodes_collected": 1,
            "episodes_with_trace_id": 1,
            "episodes_with_task_id": 1,
            "episodes_with_dataset_row": 1,
            "episodes_with_shared_generation_record": 0,
            "episodes_with_shared_trial_record": 0,
            "episodes_with_shared_trace_record": 1,
            "episodes_with_verifier_reward": 1,
        }

    def test_collect_uses_final_metrics_when_no_sidecar_exists(self, tmp_path):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        atif = {
            "schema_version": "ATIF-v1.5",
            "final_metrics": {"extra": {"reward": 0.42}},
            "steps": [
                {"step_id": 1, "source": "system", "message": "system prompt"},
                {"step_id": 2, "source": "user", "message": "write a model"},
                {
                    "step_id": 3,
                    "source": "agent",
                    "message": "here is the model",
                    "metrics": {
                        "completion_token_ids": [10, 20, 30],
                        "logprobs": [-0.1, -0.2, -0.3],
                    },
                },
            ],
        }
        (task_dir / "trajectory.json").write_text(json.dumps(atif))

        c = HarborCollector(tmp_path, shuffle=False)
        results = c.collect(episodes=1)

        assert len(results) == 1
        assert results[0].reward == pytest.approx(0.42)

    def test_collect_prefers_sidecar_reward_over_judge_by_default(self, tmp_path):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        _write_atif(task_dir / "trajectory.json", reward_total=0.9)

        class _Judge:
            def score(self, *, task_name, trajectory, artifacts):
                return Reward(total=9.9, components={"judge": 9.9}, info={"source": "judge"})

        c = HarborCollector(tmp_path, shuffle=False, judge=_Judge())
        results = c.collect(episodes=1)

        assert len(results) == 1
        assert results[0].reward == pytest.approx(0.9)

    def test_collect_can_opt_into_judge_recompute(self, tmp_path):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        _write_atif(task_dir / "trajectory.json", reward_total=0.9)

        class _Judge:
            def score(self, *, task_name, trajectory, artifacts):
                return Reward(total=9.9, components={"judge": 9.9}, info={"source": "judge"})

        c = HarborCollector(
            tmp_path,
            shuffle=False,
            judge=_Judge(),
            prefer_verifier_reward=False,
        )
        results = c.collect(episodes=1)

        assert len(results) == 1
        assert results[0].reward == pytest.approx(9.9)

    def test_resolve_reward_marks_structured_sidecar_as_verifier(self, tmp_path):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        atif_path = task_dir / "trajectory.json"
        _write_atif(atif_path, reward_total=0.9)
        (task_dir / "reward.json").write_text(
            json.dumps(
                {
                    "schema_version": "synthstats.harbor.reward.v1",
                    "reward": 0.9,
                    "reward_source": "verifier",
                    "evidence_kind": "verifier",
                    "truth_owner": "verifier",
                    "components": {"elpd": -1.0},
                    "loo_status": "accepted",
                    "diagnostics": {},
                    "info": {},
                }
            )
        )

        c = HarborCollector(tmp_path, shuffle=False)
        reward = c._resolve_reward(json.loads(atif_path.read_text()), atif_path)

        assert reward.total == pytest.approx(0.9)
        assert reward.info["reward_source"] == "verifier"
        assert reward.info["evidence_kind"] == "verifier"
        assert reward.info["truth_owner"] == "verifier"

    def test_resolve_reward_marks_plain_sidecar_as_heuristic(self, tmp_path):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        atif_path = task_dir / "trajectory.json"
        _write_atif(atif_path, reward_total=0.9)

        c = HarborCollector(tmp_path, shuffle=False)
        reward = c._resolve_reward(json.loads(atif_path.read_text()), atif_path)

        assert reward.total == pytest.approx(0.9)
        assert reward.info["reward_source"] == "sidecar"
        assert reward.info["evidence_kind"] == "heuristic"
        assert reward.info["truth_owner"] == "heuristic"

    def test_resolve_reward_marks_final_metrics_as_heuristic(self, tmp_path):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        atif = {
            "schema_version": "ATIF-v1.5",
            "final_metrics": {"extra": {"reward": 0.42}},
            "steps": [
                {"step_id": 1, "source": "system", "message": "system prompt"},
                {"step_id": 2, "source": "user", "message": "write a model"},
            ],
        }
        atif_path = task_dir / "trajectory.json"
        atif_path.write_text(json.dumps(atif))

        c = HarborCollector(tmp_path, shuffle=False)
        reward = c._resolve_reward(atif, atif_path)

        assert reward.total == pytest.approx(0.42)
        assert reward.info["reward_source"] == "final_metrics"
        assert reward.info["evidence_kind"] == "heuristic"
        assert reward.info["truth_owner"] == "heuristic"

    def test_resolve_reward_preserves_semantics_embedded_in_final_metrics(self, tmp_path):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        atif = {
            "schema_version": "ATIF-v1.6",
            "final_metrics": {
                "extra": {
                    "reward": -0.123,
                    "reward_source": "validation.semantic_score",
                    "evidence_kind": "verifier",
                    "truth_owner": "verifier",
                    "semantic_metric": "avg_holdout_logp",
                    "semantic_score": -0.123,
                }
            },
            "steps": [
                {"step_id": 1, "source": "system", "message": "system prompt"},
                {"step_id": 2, "source": "user", "message": "write a model"},
            ],
        }
        atif_path = task_dir / "trajectory.json"
        atif_path.write_text(json.dumps(atif))

        c = HarborCollector(tmp_path, shuffle=False)
        reward = c._resolve_reward(atif, atif_path)

        assert reward.total == pytest.approx(-0.123)
        assert reward.info["source"] == "final_metrics"
        assert reward.info["reward_source"] == "validation.semantic_score"
        assert reward.info["evidence_kind"] == "verifier"
        assert reward.info["truth_owner"] == "verifier"
        assert reward.info["semantic_metric"] == "avg_holdout_logp"
        assert reward.info["semantic_score"] == pytest.approx(-0.123)

    def test_resolve_reward_infers_verifier_semantics_from_validation_source(self, tmp_path):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        atif = {
            "schema_version": "ATIF-v1.6",
            "final_metrics": {
                "extra": {
                    "reward": -0.123,
                    "reward_source": "validation.semantic_score",
                    "semantic_metric": "avg_holdout_logp",
                    "semantic_score": -0.123,
                }
            },
            "steps": [
                {"step_id": 1, "source": "system", "message": "system prompt"},
                {"step_id": 2, "source": "user", "message": "write a model"},
            ],
        }
        atif_path = task_dir / "trajectory.json"
        atif_path.write_text(json.dumps(atif))

        c = HarborCollector(tmp_path, shuffle=False)
        reward = c._resolve_reward(atif, atif_path)

        assert reward.total == pytest.approx(-0.123)
        assert reward.info["source"] == "final_metrics"
        assert reward.info["reward_source"] == "validation.semantic_score"
        assert reward.info["evidence_kind"] == "verifier"
        assert reward.info["truth_owner"] == "verifier"
        assert reward.info["semantic_metric"] == "avg_holdout_logp"
        assert reward.info["semantic_score"] == pytest.approx(-0.123)

    def test_resolve_reward_preserves_top_level_metrics_from_sidecar(self, tmp_path):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        atif_path = task_dir / "trajectory.json"
        _write_atif(atif_path, reward_total=-0.123)
        (task_dir / "reward.json").write_text(
            json.dumps(
                {
                    "reward": -0.123,
                    "reward_source": "validation.semantic_score",
                    "evidence_kind": "verifier",
                    "truth_owner": "verifier",
                    "semantic_metric": "avg_holdout_logp",
                    "semantic_score": -0.123,
                }
            )
        )

        c = HarborCollector(tmp_path, shuffle=False)
        reward = c._resolve_reward(json.loads(atif_path.read_text()), atif_path)

        assert reward.total == pytest.approx(-0.123)
        assert reward.components["semantic_score"] == pytest.approx(-0.123)
        assert reward.info["reward_source"] == "validation.semantic_score"
        assert reward.info["evidence_kind"] == "verifier"
        assert reward.info["truth_owner"] == "verifier"
        assert reward.info["semantic_metric"] == "avg_holdout_logp"
        assert reward.info["semantic_score"] == pytest.approx(-0.123)

    def test_resolve_reward_infers_verifier_semantics_from_unannotated_elpd_metrics(
        self,
        tmp_path,
    ):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        atif = {
            "schema_version": "ATIF-v1.6",
            "session_id": "case-dugongs-0003",
            "final_metrics": {
                "extra": {
                    "reward": 0.89,
                    "compiles": True,
                    "elpd": -1.15,
                    "elpd_status": "ok",
                    "log_lik_shape": [200, 10],
                }
            },
            "steps": [
                {"step_id": 1, "source": "system", "message": "system prompt"},
                {"step_id": 2, "source": "user", "message": "write a model"},
            ],
        }
        atif_path = task_dir / "trajectory.json"
        atif_path.write_text(json.dumps(atif))

        c = HarborCollector(tmp_path, shuffle=False)
        reward = c._resolve_reward(atif, atif_path)

        assert reward.total == pytest.approx(0.89)
        assert reward.info["source"] == "final_metrics"
        assert reward.info["reward_source"] == "validation.elpd"
        assert reward.info["evidence_kind"] == "verifier"
        assert reward.info["truth_owner"] == "verifier"
        assert reward.info["elpd_status"] == "ok"
        assert reward.info["log_lik_shape"] == [200, 10]

    def test_resolve_reward_marks_unbackfilled_changed_programs_as_heuristic(
        self,
        tmp_path,
    ):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        atif = {
            "schema_version": "ATIF-v1.6",
            "session_id": "case-dugongs-0003",
            "final_metrics": {
                "extra": {
                    "reward": 0.89,
                    "reward_source": "validation.elpd",
                    "evidence_kind": "verifier",
                    "truth_owner": "verifier",
                    "compiles": True,
                    "elpd": -1.15,
                    "elpd_status": "failed",
                }
            },
            "steps": [
                {"step_id": 1, "source": "system", "message": "system prompt"},
                {"step_id": 2, "source": "user", "message": "write a model"},
                {
                    "step_id": 3,
                    "source": "agent",
                    "tool_calls": [
                        {
                            "function_name": "submit_program",
                            "arguments": {"code": "main = rewritten"},
                        }
                    ],
                },
            ],
        }
        backup_atif = {
            **atif,
            "steps": [
                {"step_id": 1, "source": "system", "message": "system prompt"},
                {"step_id": 2, "source": "user", "message": "write a model"},
                {
                    "step_id": 3,
                    "source": "agent",
                    "tool_calls": [
                        {
                            "function_name": "submit_program",
                            "arguments": {"code": "main = original"},
                        }
                    ],
                },
            ],
        }
        atif_path = task_dir / "trajectory.json"
        atif_path.write_text(json.dumps(atif))
        atif_path.with_suffix(".json.bak").write_text(json.dumps(backup_atif))

        c = HarborCollector(tmp_path, shuffle=False)
        reward = c._resolve_reward(atif, atif_path)

        assert reward.total == pytest.approx(0.89)
        assert reward.info["source"] == "final_metrics"
        assert reward.info["reward_source"] == "heuristic.code_change"
        assert reward.info["evidence_kind"] == "heuristic"
        assert reward.info["truth_owner"] == "heuristic"
        assert reward.info["verification_stale"] is True
        assert reward.info["verification_stale_reason"] == "program_mismatch"

    def test_trajectory_artifacts_normalize_unannotated_final_metrics(self, tmp_path):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        atif = {
            "schema_version": "ATIF-v1.6",
            "session_id": "case-dugongs-0003",
            "final_metrics": {
                "extra": {
                    "reward": 0.89,
                    "compiles": True,
                    "elpd": -1.15,
                    "elpd_status": "ok",
                    "log_lik_shape": [200, 10],
                }
            },
            "steps": [
                {"step_id": 1, "source": "system", "message": "system prompt"},
                {"step_id": 2, "source": "user", "message": "write a model"},
            ],
        }
        atif_path = task_dir / "trajectory.json"
        atif_path.write_text(json.dumps(atif))

        c = HarborCollector(tmp_path, shuffle=False)
        artifacts = c._trajectory_artifacts_from_atif(atif, atif_path=atif_path)

        final_metrics = artifacts["final_metrics"]
        assert final_metrics["reward_source"] == "validation.elpd"
        assert final_metrics["evidence_kind"] == "verifier"
        assert final_metrics["truth_owner"] == "verifier"
        assert final_metrics["source"] == "final_metrics"

    def test_trajectory_artifacts_mark_unbackfilled_changed_programs_as_stale(
        self,
        tmp_path,
    ):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        atif = {
            "schema_version": "ATIF-v1.6",
            "session_id": "case-dugongs-0003",
            "final_metrics": {
                "extra": {
                    "reward": 0.89,
                    "reward_source": "validation.elpd",
                    "evidence_kind": "verifier",
                    "truth_owner": "verifier",
                    "compiles": True,
                    "elpd": -1.15,
                    "elpd_status": "failed",
                }
            },
            "steps": [
                {"step_id": 1, "source": "system", "message": "system prompt"},
                {"step_id": 2, "source": "user", "message": "write a model"},
                {
                    "step_id": 3,
                    "source": "agent",
                    "tool_calls": [
                        {
                            "function_name": "submit_program",
                            "arguments": {"code": "main = rewritten"},
                        }
                    ],
                },
            ],
        }
        backup_atif = {
            **atif,
            "steps": [
                {"step_id": 1, "source": "system", "message": "system prompt"},
                {"step_id": 2, "source": "user", "message": "write a model"},
                {
                    "step_id": 3,
                    "source": "agent",
                    "tool_calls": [
                        {
                            "function_name": "submit_program",
                            "arguments": {"code": "main = original"},
                        }
                    ],
                },
            ],
        }
        atif_path = task_dir / "trajectory.json"
        atif_path.write_text(json.dumps(atif))
        atif_path.with_suffix(".json.bak").write_text(json.dumps(backup_atif))

        c = HarborCollector(tmp_path, shuffle=False)
        artifacts = c._trajectory_artifacts_from_atif(atif, atif_path=atif_path)

        final_metrics = artifacts["final_metrics"]
        assert final_metrics["source"] == "final_metrics"
        assert final_metrics["reward_source"] == "heuristic.code_change"
        assert final_metrics["evidence_kind"] == "heuristic"
        assert final_metrics["truth_owner"] == "heuristic"
        assert final_metrics["verification_stale"] is True
        assert final_metrics["verification_stale_reason"] == "program_mismatch"

    def test_resolve_reward_marks_judge_recompute_as_heuristic(self, tmp_path):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        atif = {
            "schema_version": "ATIF-v1.5",
            "final_metrics": {"extra": {}},
            "steps": [
                {"step_id": 1, "source": "system", "message": "system prompt"},
                {"step_id": 2, "source": "user", "message": "write a model"},
            ],
        }
        atif_path = task_dir / "trajectory.json"
        atif_path.write_text(json.dumps(atif))

        class _Judge:
            def score(self, *, task_name, trajectory, artifacts):
                return Reward(total=0.7, components={"judge": 0.7}, info={"source": "judge"})

        c = HarborCollector(
            tmp_path,
            shuffle=False,
            judge=_Judge(),
            prefer_verifier_reward=False,
        )
        reward = c._resolve_reward(atif, atif_path)

        assert reward.total == pytest.approx(0.7)
        assert reward.info["reward_source"] == "judge"
        assert reward.info["evidence_kind"] == "heuristic"
        assert reward.info["truth_owner"] == "heuristic"
