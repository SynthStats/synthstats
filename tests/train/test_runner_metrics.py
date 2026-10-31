from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from synthstats.train.runners.metrics import extract_trainer_metrics


class TestRunnerMetrics:
    def test_extract_trainer_metrics_returns_empty_for_none(self):
        assert extract_trainer_metrics(None) == {}

    def test_extract_trainer_metrics_collects_numeric_metrics_and_logz(self):
        trainer = SimpleNamespace(
            all_metrics={
                "rollout_n_success": 4,
                "rollout_n_with_trace_id": 3.0,
                "reward_sources": ["verifier"],
                "has_verifier": True,
            },
            logZ=torch.tensor(2.5),
        )

        assert extract_trainer_metrics(trainer) == {
            "rollout_n_success": pytest.approx(4.0),
            "rollout_n_with_trace_id": pytest.approx(3.0),
            "logZ": pytest.approx(2.5),
        }
