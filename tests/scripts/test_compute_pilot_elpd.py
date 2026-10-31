from __future__ import annotations

import json

import compute_pilot_elpd as script
import numpy as np
import pytest
from conftest import build_sft_record

from synthstats.executors.lazyppl_sandbox import LazyPPLExecutor


def test_in_sample_log_predictive_density_averages_observation_columns():
    log_lik = np.log(np.array([[0.25, 0.5], [0.75, 0.5]]))

    score = script.compute_in_sample_log_predictive_density(log_lik)

    assert score == pytest.approx(-0.6931471805599453)


def test_output_names_the_in_sample_predictive_quantity(tmp_path):
    path = tmp_path / "pilot-dugongs-0000.json"
    record = build_sft_record("")
    record["final_metrics"]["extra"].update({"elpd": 1.0, "elpd_status": "ok"})
    path.write_text(json.dumps(record))

    assert script.main(atif_dir=tmp_path, executor=LazyPPLExecutor(lazyppl_dir=str(tmp_path))) == 1

    metrics = json.loads(path.read_text())["final_metrics"]["extra"]
    assert "elpd" not in metrics
    assert metrics["in_sample_log_predictive_density"] is None
    assert metrics["in_sample_log_predictive_density_status"] == "missing_code"


def test_main_returns_error_code_when_no_input_files(tmp_path):
    atif_dir = tmp_path / "atif_trajectories"
    atif_dir.mkdir()

    assert script.main(atif_dir=atif_dir) == 1


def test_main_writes_reward_metadata(tmp_path):
    atif_dir = tmp_path / "atif_trajectories"
    pilot_data_dir = tmp_path / "pilot_data"
    atif_dir.mkdir()
    pilot_data_dir.mkdir()

    atif_path = atif_dir / "pilot-dugongs-0001.json"
    atif_path.write_text(
        json.dumps(build_sft_record("module Submission where\nimport LazyPPL")),
        encoding="utf-8",
    )
    (pilot_data_dir / "lazyppl_pilot_dugongs.json").write_text(
        json.dumps(
            {
                "columns": [{"name": "age", "hs_type": "Double"}],
                "observations": [{"age": 1.0}, {"age": 2.0}],
            }
        ),
        encoding="utf-8",
    )

    class FakeExecutor:
        def execute_for_loglik(
            self,
            code,
            columns,
            data_rows,
            *,
            burnin,
            thinning,
            ndraws,
            timeout_s,
        ):
            assert code == "module Submission where\nimport LazyPPL"
            assert columns == [{"name": "age", "hs_type": "Double"}]
            assert data_rows == [{"age": 1.0}, {"age": 2.0}]
            assert burnin == script.BURNIN
            assert thinning == script.THINNING
            assert ndraws == script.NDRAWS
            assert timeout_s == script.TIMEOUT_S
            return np.full((2, 2), -1.0)

    assert (
        script.main(
            atif_dir=atif_dir,
            pilot_data_dir=pilot_data_dir,
            executor=FakeExecutor(),
        )
        == 0
    )

    payload = json.loads(atif_path.read_text())
    metrics = payload["final_metrics"]["extra"]

    assert metrics["in_sample_log_predictive_density_status"] == "ok"
    assert metrics["log_lik_shape"] == [2, 2]
    assert metrics["in_sample_log_predictive_density"] == pytest.approx(-1.0)
    assert metrics["reward"] == pytest.approx(0.9048374180359595)
    assert metrics["reward_source"] == "validation.in_sample_log_predictive_density"
    assert metrics["evidence_kind"] == "verifier"
    assert metrics["truth_owner"] == "verifier"


def test_main_marks_missing_code_as_partial_failure(tmp_path):
    atif_dir = tmp_path / "atif_trajectories"
    atif_dir.mkdir()
    atif_path = atif_dir / "pilot-dugongs-0001.json"
    atif_path.write_text(
        json.dumps(build_sft_record("")),
        encoding="utf-8",
    )

    assert script.main(atif_dir=atif_dir, executor=object()) == 1

    payload = json.loads(atif_path.read_text())
    metrics = payload["final_metrics"]["extra"]
    assert metrics["reward"] == script.MIN_REWARD
    assert metrics["in_sample_log_predictive_density"] is None
    assert metrics["in_sample_log_predictive_density_status"] == "missing_code"
    assert metrics["reward_source"] == "verifier_missing_program"
    assert metrics["evidence_kind"] == "verifier"
    assert metrics["truth_owner"] == "verifier"
