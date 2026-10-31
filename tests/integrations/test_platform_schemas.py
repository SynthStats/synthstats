from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from synthstats.core import platform_adapters as adapters
from synthstats.core.platform_adapters import (
    evidence_card_from_records,
    evidence_record_from_reward_artifact,
)
from synthstats.core.platform_schemas import (
    EvidenceCard,
    EvidenceRecord,
    GenerationRecord,
    RuntimeDescriptor,
    TrialRecord,
)
from synthstats.core.types import Reward
from synthstats.integrations.harbor.reward_artifacts import build_reward_artifact


def _platform_contracts() -> SimpleNamespace:
    return SimpleNamespace(
        EvidenceCard=EvidenceCard,
        EvidenceRecord=EvidenceRecord,
        GenerationRecord=GenerationRecord,
        RuntimeDescriptor=RuntimeDescriptor,
        TrialRecord=TrialRecord,
        evidence_card_from_records=evidence_card_from_records,
        evidence_record_from_reward_artifact=evidence_record_from_reward_artifact,
    )


def test_reward_artifact_maps_to_shared_evidence_record() -> None:
    contracts = _platform_contracts()
    artifact = build_reward_artifact(
        Reward(
            total=0.75,
            components={"elpd": -3.0},
            info={"loo_status": "accepted", "diagnostics": {"warning": False}},
        ),
        reward_source="validation.elpd",
        provenance={"task_name": "dugongs", "language": "pymc"},
    )

    evidence = contracts.evidence_record_from_reward_artifact(
        artifact,
        trial_id="trial-001",
    )

    assert "reward_source" not in artifact["info"]
    assert "evidence_kind" not in artifact["info"]
    assert "truth_owner" not in artifact["info"]
    assert evidence.trial_id == "trial-001"
    assert evidence.task_id == "dugongs"
    assert evidence.kind == "verifier"
    assert evidence.truth_owner == "verifier"
    assert evidence.status == "accepted"
    assert evidence.value == 0.75
    assert evidence.components == {"elpd": -3.0}
    assert "truth_owner" not in evidence.metadata


def test_evidence_record_does_not_promote_truth_owner_from_metadata() -> None:
    contracts = _platform_contracts()

    evidence = contracts.EvidenceRecord.model_validate(
        {
            "evidence_id": "evidence-001",
            "source": "validation.elpd",
            "kind": "verifier",
            "status": "accepted",
            "value": 0.75,
            "metadata": {
                "truth_owner": "verifier",
                "evidence_note": "prior",
            },
        }
    )

    assert evidence.truth_owner is None
    assert evidence.metadata["evidence_note"] == "prior"


def test_evidence_record_requires_canonical_kind_field() -> None:
    contracts = _platform_contracts()

    with pytest.raises(ValidationError):
        contracts.EvidenceRecord.model_validate(
            {
                "evidence_id": "evidence-kind-001",
                "source": "validation.elpd",
                "evidence_kind": "verifier",
                "status": "accepted",
                "value": 0.75,
                "metadata": {"truth_owner": "verifier"},
            }
        )


def test_evidence_record_requires_top_level_kind() -> None:
    contracts = _platform_contracts()

    with pytest.raises(ValidationError):
        contracts.EvidenceRecord.model_validate(
            {
                "evidence_id": "evidence-meta-kind-001",
                "source": "validation.elpd",
                "status": "accepted",
                "value": 0.75,
                "metadata": {
                    "evidence_kind": "verifier",
                    "truth_owner": "verifier",
                },
            }
        )


def test_agent_result_and_messages_map_to_shared_trial_and_trace_records() -> None:
    agent_result = {
        "program": "x = 1",
        "language": "pymc",
        "reward": 0.5,
        "task_config": {"env": {"name": "dugongs"}},
    }

    trial = adapters.trial_record_from_agent_result(
        agent_result,
        trial_id="trial-001",
        trace_id="trace-001",
    )
    trace = adapters.trace_record_from_messages(
        [
            {
                "role": "user",
                "content": "Solve the task.",
                "metadata": {
                    "phase": "prompt",
                    "evidence_kind": "heuristic",
                    "truth_owner": "heuristic",
                    "semantic_status": "not_scored",
                },
            },
            {
                "role": "assistant",
                "content": "I will fit a PyMC model.",
                "evidence_kind": "verifier",
                "truth_owner": "verifier",
                "semantic_status": "pass",
                "metadata": {"phase": "semantic_validation"},
            },
        ],
        trace_id="trace-001",
        task_id=trial.task_id,
        trial_id=trial.trial_id,
        artifacts={"program": agent_result["program"]},
    )

    assert trial.task_id == "dugongs"
    assert trial.runtime is not None
    assert trial.runtime.name == "pymc"
    assert trial.trace_id == "trace-001"
    assert trace.task_id == "dugongs"
    assert trace.trial_id == "trial-001"
    assert [step.role for step in trace.steps] == ["user", "assistant"]
    assert trace.steps[0].evidence_kind == "heuristic"
    assert trace.steps[0].truth_owner == "heuristic"
    assert trace.steps[0].semantic_status == "not_scored"
    assert trace.steps[0].metadata["phase"] == "prompt"
    assert trace.steps[1].evidence_kind == "verifier"
    assert trace.steps[1].truth_owner == "verifier"
    assert trace.steps[1].semantic_status == "pass"
    assert trace.steps[1].metadata["phase"] == "semantic_validation"


def test_dataset_dataset_row_maps_to_shared_trial_and_trace_records() -> None:
    dataset_row = {
        "id": "completion-001",
        "prompt_id": "prompt-001",
        "env_id": "dugongs",
        "split": "train",
        "prompt": "Write a model for grouped counts.",
        "completion": "model code",
        "model_type": "hierarchical",
        "domain": "ecology",
        "complexity": "medium",
        "key_concepts": ["counts"],
        "has_observed_data": True,
        "pymc_version": "5.20.0",
        "trace_id": "completion-001:validation",
        "candidate_outputs_json": json.dumps(["```python\nx = 2\n```", "```python\nx = 3\n```"]),
        "provenance_json": json.dumps(
            {
                "id": "completion-001",
                "prompt_id": "prompt-001",
                "provenance": {"runtime_name": "pymc", "runtime_version": "5.20.0"},
                "metadata": {},
            }
        ),
        "env_name": "Dugongs",
        "env_spec_json": "{}",
        "data_ref": "data/dugongs.parquet",
        "seed": 7,
        "generator_version": "0.1.0",
        "trace_json": json.dumps(
            {
                "schema_version": "ppl.platform.record.v0",
                "trace_id": "completion-001:validation",
                "task_id": "dugongs",
                "trial_id": "completion-001",
                "steps": [
                    {
                        "step_id": "step-0000",
                        "role": "validator",
                        "content": "ok",
                        "evidence_kind": "verifier",
                        "truth_owner": "verifier",
                        "semantic_status": "not_scored",
                        "metadata": {
                            "attempt": 0,
                            "phase": "runtime",
                            "ok": True,
                            "evidence_kind": "verifier",
                            "truth_owner": "verifier",
                            "semantic_status": "not_scored",
                        },
                    }
                ],
                "artifacts": {
                    "outcome": "validated",
                    "prompt_id": "prompt-001",
                    "split": "train",
                },
                "metadata": {
                    "runtime": "pymc",
                    "env_id": "dugongs",
                    "split": "train",
                },
            }
        ),
        "validation_json": json.dumps({"ok": True, "message": "passed"}),
    }

    generation = adapters.generation_record_from_dataset_row(dataset_row)
    trial = adapters.trial_record_from_dataset_row(dataset_row)
    trace = adapters.trace_record_from_dataset_row(dataset_row)

    assert generation.record_id == "completion-001"
    assert generation.runtime is not None
    assert generation.runtime.name == "pymc"
    assert generation.prompt is not None
    assert generation.prompt.view_id == "prompt-001"
    assert generation.candidate_outputs == [
        "```python\nx = 2\n```",
        "```python\nx = 3\n```",
    ]
    assert generation.validation["ok"] is True
    assert trial.trial_id == "completion-001"
    assert trial.task_id == "dugongs"
    assert trial.runtime is not None
    assert trial.runtime.name == "pymc"
    assert trial.trace_id == "completion-001:validation"
    assert trial.artifacts["validation"]["ok"] is True
    assert trace is not None
    assert trace.trace_id == "completion-001:validation"
    assert trace.trial_id == "completion-001"
    assert trace.steps[0].evidence_kind == "verifier"
    assert trace.steps[0].truth_owner == "verifier"
    assert trace.steps[0].semantic_status == "not_scored"
    assert trace.steps[0].metadata["evidence_kind"] == "verifier"


def test_operator_evidence_card_prefers_verifier_reward_over_semantic_metric() -> None:
    contracts = _platform_contracts()
    dataset_row = {
        "id": "completion-001",
        "prompt_id": "prompt-001",
        "env_id": "statistical-modeling",
        "split": "train",
        "prompt": "Fit a hierarchical PyMC model.",
        "completion": "model code",
        "pymc_version": "5.20.0",
        "trace_id": "completion-001:validation",
        "candidate_outputs_json": json.dumps(["alternate model"]),
        "provenance_json": json.dumps(
            {
                "id": "completion-001",
                "prompt_id": "prompt-001",
                "provenance": {"runtime_name": "pymc", "runtime_version": "5.20.0"},
                "metadata": {},
            }
        ),
        "env_name": "Grouped Count Modeling",
        "data_ref": "data/grouped_counts.parquet",
        "trace_json": json.dumps(
            {
                "schema_version": "ppl.platform.record.v0",
                "trace_id": "completion-001:validation",
                "task_id": "statistical-modeling",
                "trial_id": "completion-001",
                "steps": [
                    {
                        "step_id": "step-0000",
                        "role": "validator",
                        "content": "ok",
                        "metadata": {"phase": "runtime", "ok": True},
                    }
                ],
                "artifacts": {"outcome": "validated"},
                "metadata": {"runtime": "pymc"},
            }
        ),
        "validation_json": json.dumps(
            {
                "ok": True,
                "message": "passed",
                "repaired": False,
                "evidence_kind": "verifier",
                "truth_owner": "verifier",
                "semantic_status": "pass",
                "semantic_metric": "psis_loo",
                "semantic_score": 0.81,
                "semantic_details": {"pareto_k_max": 0.42},
            }
        ),
    }

    generation = adapters.generation_record_from_dataset_row(dataset_row)
    trial = adapters.trial_record_from_dataset_row(dataset_row)
    trace = adapters.trace_record_from_dataset_row(dataset_row)
    artifact = build_reward_artifact(
        Reward(
            total=0.93,
            components={"elpd": -1.2},
            info={"loo_status": "accepted", "diagnostics": {"warning": False}},
        ),
        reward_source="validation.elpd",
        provenance={"task_name": "statistical-modeling", "language": "pymc"},
    )
    evidence = contracts.evidence_record_from_reward_artifact(
        artifact,
        trial_id=trial.trial_id,
    )

    card = contracts.evidence_card_from_records(
        generation=generation,
        trial=trial,
        trace=trace,
        evidence_records=[evidence],
    )

    assert card.task_id == "statistical-modeling"
    assert card.trial_id == "completion-001"
    assert card.generation_id == "completion-001"
    assert card.trace_id == "completion-001:validation"
    assert card.runtime is not None
    assert card.runtime.name == "pymc"
    assert card.candidate_status == "ready_for_review"
    assert card.operator_status == "pending_review"
    assert card.primary_metric is not None
    assert card.primary_metric.name == "reward"
    assert card.primary_metric.value == 0.93
    assert card.supporting_metrics[0].name == "psis_loo"
    assert card.recommended_action == "review"
    assert card.artifacts["candidate_output_count"] == 1
    assert card.artifacts["trace_step_count"] == 1


def test_operator_evidence_card_prefers_verifier_reward_over_heuristic_order() -> None:
    contracts = _platform_contracts()
    runtime = contracts.RuntimeDescriptor(
        name="pymc",
        version="5.20.0",
        execution_backend="python",
    )
    generation = contracts.GenerationRecord(
        record_id="generation-heuristic-first",
        task_id="statistical-modeling",
        runtime=runtime,
        raw_output="model code",
        validation={
            "ok": True,
            "semantic_status": "pass",
            "semantic_metric": "psis_loo",
            "semantic_score": 0.81,
            "evidence_kind": "verifier",
            "truth_owner": "verifier",
        },
    )
    trial = contracts.TrialRecord(
        trial_id="trial-heuristic-first",
        task_id="statistical-modeling",
        runtime=runtime,
        program="model code",
    )
    heuristic = contracts.EvidenceRecord(
        evidence_id="evidence-heuristic",
        task_id="statistical-modeling",
        trial_id="trial-heuristic-first",
        source="judge",
        kind="heuristic",
        truth_owner="heuristic",
        status="accepted",
        metric_name="judge_score",
        value=0.99,
    )
    verifier = contracts.EvidenceRecord(
        evidence_id="evidence-verifier",
        task_id="statistical-modeling",
        trial_id="trial-heuristic-first",
        source="verifier",
        kind="verifier",
        truth_owner="verifier",
        status="accepted",
        metric_name="reward",
        value=0.93,
    )

    card = contracts.evidence_card_from_records(
        generation=generation,
        trial=trial,
        evidence_records=[heuristic, verifier],
    )

    assert card.candidate_status == "ready_for_review"
    assert card.recommended_action == "review"
    assert card.primary_metric is not None
    assert card.primary_metric.name == "reward"
    assert card.primary_metric.source == "verifier"
    assert card.primary_metric.truth_owner == "verifier"
    assert {metric.name for metric in card.supporting_metrics} == {
        "judge_score",
        "psis_loo",
    }


def test_operator_evidence_card_with_only_heuristic_success_needs_verification() -> None:
    contracts = _platform_contracts()
    runtime = contracts.RuntimeDescriptor(
        name="pymc",
        version="5.20.0",
        execution_backend="python",
    )
    generation = contracts.GenerationRecord(
        record_id="generation-needs-verification",
        task_id="statistical-modeling",
        runtime=runtime,
        raw_output="model code",
    )
    trial = contracts.TrialRecord(
        trial_id="trial-needs-verification",
        task_id="statistical-modeling",
        runtime=runtime,
        program="model code",
    )
    heuristic = contracts.EvidenceRecord(
        evidence_id="evidence-heuristic-only",
        task_id="statistical-modeling",
        trial_id="trial-needs-verification",
        source="judge",
        kind="heuristic",
        truth_owner="heuristic",
        status="accepted",
        metric_name="judge_score",
        value=0.88,
    )

    card = contracts.evidence_card_from_records(
        generation=generation,
        trial=trial,
        evidence_records=[heuristic],
    )

    assert card.candidate_status == "needs_verification"
    assert card.recommended_action == "run_verifier"
    assert card.primary_metric is not None
    assert card.primary_metric.name == "judge_score"
    assert card.primary_metric.truth_owner == "heuristic"


def test_operator_evidence_card_ignores_heuristic_reject_when_verifier_passes() -> None:
    contracts = _platform_contracts()
    runtime = contracts.RuntimeDescriptor(
        name="pymc",
        version="5.20.0",
        execution_backend="python",
    )
    generation = contracts.GenerationRecord(
        record_id="generation-verifier-wins",
        task_id="statistical-modeling",
        runtime=runtime,
        raw_output="model code",
        validation={
            "ok": True,
            "semantic_status": "pass",
            "semantic_metric": "psis_loo",
            "semantic_score": 0.81,
            "evidence_kind": "verifier",
            "truth_owner": "verifier",
        },
    )
    trial = contracts.TrialRecord(
        trial_id="trial-verifier-wins",
        task_id="statistical-modeling",
        runtime=runtime,
        program="model code",
    )
    heuristic_reject = contracts.EvidenceRecord(
        evidence_id="evidence-heuristic-reject",
        task_id="statistical-modeling",
        trial_id="trial-verifier-wins",
        source="judge",
        kind="heuristic",
        truth_owner="heuristic",
        status="rejected",
        metric_name="judge_score",
        value=0.01,
    )
    verifier = contracts.EvidenceRecord(
        evidence_id="evidence-verifier-pass",
        task_id="statistical-modeling",
        trial_id="trial-verifier-wins",
        source="verifier",
        kind="verifier",
        truth_owner="verifier",
        status="accepted",
        metric_name="reward",
        value=0.91,
    )

    card = contracts.evidence_card_from_records(
        generation=generation,
        trial=trial,
        evidence_records=[heuristic_reject, verifier],
    )

    assert card.candidate_status == "ready_for_review"
    assert card.recommended_action == "review"
    assert card.primary_metric is not None
    assert card.primary_metric.name == "reward"
