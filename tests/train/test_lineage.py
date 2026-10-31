from __future__ import annotations

import torch

from synthstats.train.lineage import (
    apply_lineage_batching,
    compute_lineage_batch_order,
    compute_lineage_loss_weight,
    compute_lineage_loss_weights,
    extract_numeric_rollout_metrics,
    extract_rollout_metric_sequences,
)


class TestLineage:
    def test_extract_numeric_rollout_metrics_from_rollout_metrics_attr(self):
        class FakeBatch:
            rollout_metrics = {
                "n_success": 2,
                "n_with_trace_id": 1,
                "reward_sources": ["verifier"],
                "reward_evidence_kinds": ["verifier"],
                "reward_truth_owners": ["verifier"],
            }

        assert extract_numeric_rollout_metrics(FakeBatch()) == {
            "n_success": 2.0,
            "n_with_trace_id": 1.0,
            "n_reward_evidence_kind_verifier": 1.0,
            "reward_evidence_kind_verifier_fraction": 1.0,
            "n_reward_truth_owner_verifier": 1.0,
            "reward_truth_owner_verifier_fraction": 1.0,
        }

    def test_extract_rollout_metric_sequences_from_rollout_metrics_attr(self):
        class FakeBatch:
            rollout_metrics = {
                "has_verifier_reward": [1, 0, 1],
                "reward_sources": ["verifier", "judge", "verifier"],
            }

        assert extract_rollout_metric_sequences(FakeBatch()) == {
            "has_verifier_reward": [1.0, 0.0, 1.0],
        }

    def test_compute_lineage_loss_weights_uses_rollout_flags(self):
        class FakeConfig:
            algorithm = {
                "lineage_weighting": {
                    "metric": "n_with_verifier_reward",
                    "denominator_metric": "n_success",
                    "base_weight": 1.0,
                    "scale": 0.5,
                    "min_weight": 1.0,
                    "max_weight": 2.0,
                }
            }

        class FakeBatch:
            rollout_metrics = {
                "n_success": 3,
                "n_with_verifier_reward": 2,
                "has_verifier_reward": [1, 0, 1],
            }

        assert compute_lineage_loss_weights(FakeBatch(), FakeConfig()) == [1.5, 1.0, 1.5]

    def test_compute_lineage_loss_weight_averages_per_rollout_weights(self):
        class FakeConfig:
            algorithm = {
                "lineage_weighting": {
                    "metric": "n_with_verifier_reward",
                    "denominator_metric": "n_success",
                    "base_weight": 1.0,
                    "scale": 0.5,
                }
            }

        class FakeBatch:
            rollout_metrics = {
                "n_success": 3,
                "n_with_verifier_reward": 2,
                "has_verifier_reward": [1, 0, 1],
            }

        assert compute_lineage_loss_weight(FakeBatch(), FakeConfig()) == 4.0 / 3.0

    def test_compute_lineage_batch_order_groups_positive_rollouts_first(self):
        class FakeConfig:
            algorithm = {
                "lineage_batching": {
                    "metric": "n_with_verifier_reward",
                    "positive_first": True,
                }
            }

        class FakeBatch:
            rollout_metrics = {
                "has_verifier_reward": [0, 1, 0, 1],
            }

        assert compute_lineage_batch_order(FakeBatch(), FakeConfig()) == [1, 3, 0, 2]

    def test_apply_lineage_batching_reorders_tensors_and_rollout_metadata(self):
        class FakeConfig:
            algorithm = {
                "lineage_batching": {
                    "metric": "n_with_verifier_reward",
                    "positive_first": True,
                }
            }

        class FakeBatch:
            response_ids = torch.tensor([[10], [20], [30], [40]])
            rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
            non_tensor_batch = {
                "rollout_metrics": {
                    "n_success": 4,
                    "n_with_verifier_reward": 2,
                    "has_verifier_reward": [0, 1, 0, 1],
                    "reward_sources": [
                        "heuristic",
                        "verifier",
                        "heuristic",
                        "verifier",
                    ],
                    "reward_evidence_kinds": [
                        "heuristic",
                        "verifier",
                        "heuristic",
                        "verifier",
                    ],
                    "reward_truth_owners": [
                        "heuristic",
                        "verifier",
                        "heuristic",
                        "verifier",
                    ],
                }
            }

        batch = FakeBatch()
        metrics = apply_lineage_batching(batch, FakeConfig())

        assert batch.response_ids.squeeze(-1).tolist() == [20, 40, 10, 30]
        assert batch.rewards.tolist() == [2.0, 4.0, 1.0, 3.0]
        assert batch.non_tensor_batch["rollout_metrics"]["has_verifier_reward"] == [
            1,
            1,
            0,
            0,
        ]
        assert batch.non_tensor_batch["rollout_metrics"]["reward_sources"] == [
            "verifier",
            "verifier",
            "heuristic",
            "heuristic",
        ]
        assert batch.non_tensor_batch["rollout_metrics"]["reward_evidence_kinds"] == [
            "verifier",
            "verifier",
            "heuristic",
            "heuristic",
        ]
        assert batch.non_tensor_batch["rollout_metrics"]["reward_truth_owners"] == [
            "verifier",
            "verifier",
            "heuristic",
            "heuristic",
        ]
        assert metrics["lineage_batch_reordered"] == 1.0
        assert metrics["lineage_batch_positive_count"] == 2.0
        assert metrics["lineage_batch_negative_count"] == 2.0
        assert metrics["lineage_batch_positive_first"] == 1.0
