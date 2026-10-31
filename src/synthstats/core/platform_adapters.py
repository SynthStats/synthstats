from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from synthstats.core.platform_schemas import (
    EvidenceCard,
    EvidenceMetric,
    EvidenceRecord,
    GenerationRecord,
    PromptView,
    RuntimeDescriptor,
    TraceRecord,
    TraceStep,
    TrialRecord,
)
from synthstats.core.runtime_registry import (
    get_runtime_spec,
    normalize_runtime_name,
    runtime_name_from_payload,
    runtime_version_from_metadata,
)


def _string(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _float_map(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, raw in value.items():
        numeric = _number(raw)
        if numeric is not None:
            out[str(key)] = numeric
    return out


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _json_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _drop_none_values(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in payload.items() if value is not None}


def runtime_descriptor_from_dataset_row(
    row: Mapping[str, Any],
) -> RuntimeDescriptor | None:
    prov_wrapper = _json_dict(row.get("provenance_json"))
    prov = _dict(prov_wrapper.get("provenance"))
    runtime_name = normalize_runtime_name(
        _string(prov.get("runtime_name")) or runtime_name_from_payload(row)
    )
    if runtime_name is None:
        return None
    spec = get_runtime_spec(runtime_name)
    version = (
        _string(prov.get("runtime_version"))
        or runtime_version_from_metadata(row, spec.name)
        or spec.default_version
    )
    return RuntimeDescriptor(
        name=spec.name,
        version=version,
        execution_backend=spec.execution_backend,
    )


def generation_record_from_dataset_row(row: Mapping[str, Any]) -> GenerationRecord:
    record_id = _string(row.get("id")) or _string(row.get("record_id")) or "generation"
    task_id = _string(row.get("env_id")) or _string(row.get("task_id"))
    prompt_id = _string(row.get("prompt_id"))
    split = _string(row.get("split"))
    runtime = runtime_descriptor_from_dataset_row(row)
    prompt_text = str(row.get("prompt") or "")
    prompt_view = (
        PromptView(
            view_id=prompt_id or f"{record_id}:prompt",
            task_id=task_id,
            split=split,
            content=prompt_text,
        )
        if prompt_text or prompt_id
        else None
    )

    candidate_outputs: list[str] = []
    raw_candidates = row.get("candidate_outputs_json")
    if isinstance(raw_candidates, str) and raw_candidates:
        try:
            parsed_candidates = json.loads(raw_candidates)
        except json.JSONDecodeError:
            parsed_candidates = None
        if isinstance(parsed_candidates, list):
            candidate_outputs = [str(item) for item in parsed_candidates]
    elif isinstance(row.get("candidate_outputs"), Sequence) and not isinstance(
        row.get("candidate_outputs"), str | bytes
    ):
        candidate_outputs = [str(item) for item in row["candidate_outputs"]]

    validation = _json_dict(row.get("validation_json")) or _dict(row.get("validation"))
    metadata = _drop_none_values(
        {
            "env_name": _string(row.get("env_name")),
            "data_ref": _string(row.get("data_ref")),
            "trace_id": _string(row.get("trace_id")),
        }
    )
    return GenerationRecord(
        record_id=record_id,
        task_id=task_id,
        prompt_id=prompt_id,
        split=split,
        runtime=runtime,
        prompt=prompt_view,
        raw_output=str(row.get("completion") or ""),
        candidate_outputs=candidate_outputs,
        validation=validation,
        metadata=metadata,
    )


def trial_record_from_dataset_row(row: Mapping[str, Any]) -> TrialRecord:
    trial_id = _string(row.get("id")) or _string(row.get("trial_id")) or "trial"
    task_id = _string(row.get("env_id")) or _string(row.get("task_id"))
    runtime = runtime_descriptor_from_dataset_row(row)
    language = runtime.name if runtime is not None else _string(row.get("language"))
    validation = _json_dict(row.get("validation_json")) or _dict(row.get("validation"))
    reward = _number(row.get("reward"))
    if reward is None:
        reward = _number(validation.get("semantic_score"))
    artifacts = _drop_none_values(
        {
            "prompt_id": _string(row.get("prompt_id")),
            "split": _string(row.get("split")),
            "env_name": _string(row.get("env_name")),
            "data_ref": _string(row.get("data_ref")),
        }
    )
    if validation:
        artifacts["validation"] = validation
    return TrialRecord(
        trial_id=trial_id,
        task_id=task_id,
        language=language,
        program=_string(row.get("program")) or _string(row.get("completion")),
        reward=reward,
        runtime=runtime,
        trace_id=_string(row.get("trace_id")),
        artifacts=artifacts,
        metadata={"source": "synthstats.dataset_row"},
    )


def _steps_from_validation_trace(raw_steps: Sequence[Any]) -> list[TraceStep]:
    steps: list[TraceStep] = []
    for idx, item in enumerate(raw_steps):
        if not isinstance(item, Mapping):
            continue
        step_meta = _dict(item.get("metadata"))
        for key in ("attempt", "phase", "ok"):
            if key in item and key not in step_meta:
                step_meta[key] = item[key]
        steps.append(
            TraceStep(
                step_id=str(item.get("step_id") or f"step-{idx:04d}"),
                role=str(item.get("role") or "validator"),
                content=str(item.get("content") or item.get("message") or ""),
                evidence_kind=_string(item.get("evidence_kind")),
                truth_owner=_string(item.get("truth_owner")),
                semantic_status=_string(item.get("semantic_status")),
                metadata=step_meta,
            )
        )
    return steps


def trace_record_from_dataset_row(row: Mapping[str, Any]) -> TraceRecord | None:
    raw_trace = _json_dict(row.get("trace_json"))
    validation = _json_dict(row.get("validation_json")) or _dict(row.get("validation"))
    val_trace = _dict(validation.get("trace"))
    raw_val_steps = val_trace.get("steps")
    val_steps = raw_val_steps if isinstance(raw_val_steps, list) else []

    if raw_trace:
        trace = TraceRecord.model_validate(raw_trace)
        if not trace.steps and val_steps:
            trace.steps = _steps_from_validation_trace(val_steps)
            trace.metadata["trace_origin"] = "trace_json.empty+validation_json.trace"
        return trace

    if val_steps:
        trial_id = _string(row.get("id")) or _string(row.get("trial_id"))
        trace_id = _string(row.get("trace_id")) or (
            f"{trial_id}:validation" if trial_id else "trace"
        )
        task_id = _string(row.get("env_id")) or _string(row.get("task_id"))
        return TraceRecord(
            trace_id=trace_id,
            task_id=task_id,
            trial_id=trial_id,
            steps=_steps_from_validation_trace(val_steps),
            metadata={"trace_origin": "validation_json.trace"},
        )

    return None


def trial_record_from_agent_result(
    agent_result: Mapping[str, Any],
    *,
    trial_id: str,
    trace_id: str | None = None,
    task_id: str | None = None,
) -> TrialRecord:
    task_cfg = _dict(agent_result.get("task_config"))
    env_cfg = _dict(task_cfg.get("env"))
    resolved_task_id = (
        task_id or _string(agent_result.get("task_id")) or _string(env_cfg.get("name"))
    )
    runtime_name = normalize_runtime_name(
        _string(agent_result.get("language"))
        or _string(agent_result.get("backend"))
        or runtime_name_from_payload(agent_result)
    )
    runtime: RuntimeDescriptor | None = None
    if runtime_name is not None:
        spec = get_runtime_spec(runtime_name)
        version = runtime_version_from_metadata(agent_result, spec.name) or spec.default_version
        runtime = RuntimeDescriptor(
            name=spec.name,
            version=version,
            execution_backend=spec.execution_backend,
        )
    return TrialRecord(
        trial_id=trial_id,
        task_id=resolved_task_id,
        language=runtime.name if runtime is not None else _string(agent_result.get("language")),
        program=_string(agent_result.get("program")) or _string(agent_result.get("final_code")),
        reward=_number(agent_result.get("reward")),
        runtime=runtime,
        trace_id=trace_id,
    )


def trace_record_from_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    trace_id: str,
    task_id: str | None = None,
    trial_id: str | None = None,
    artifacts: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TraceRecord:
    steps: list[TraceStep] = []
    for idx, msg in enumerate(messages):
        step_meta = _dict(msg.get("metadata"))
        evidence_kind = _string(msg.get("evidence_kind")) or _string(step_meta.get("evidence_kind"))
        truth_owner = _string(msg.get("truth_owner")) or _string(step_meta.get("truth_owner"))
        semantic_status = _string(msg.get("semantic_status")) or _string(
            step_meta.get("semantic_status")
        )
        steps.append(
            TraceStep(
                step_id=str(msg.get("step_id") or f"step-{idx:04d}"),
                role=str(msg.get("role") or "user"),
                content=str(msg.get("content") or ""),
                tool_call_id=_string(msg.get("tool_call_id")),
                reasoning_content=_string(msg.get("reasoning_content")),
                evidence_kind=evidence_kind,
                truth_owner=truth_owner,
                semantic_status=semantic_status,
                metadata=step_meta,
            )
        )
    return TraceRecord(
        trace_id=trace_id,
        task_id=task_id,
        trial_id=trial_id,
        steps=steps,
        artifacts=_dict(artifacts),
        metadata=_dict(metadata),
    )


def evidence_record_from_reward_artifact(
    artifact: Mapping[str, Any],
    *,
    trial_id: str | None = None,
) -> EvidenceRecord:
    provenance = _dict(artifact.get("provenance"))
    reward_source = artifact.get("reward_source")
    if not reward_source:
        import warnings

        warnings.warn(
            "reward artifact missing reward_source: annotate at creation",
            UserWarning,
            stacklevel=2,
        )
        reward_source = "unknown"
    reward_source = str(reward_source)
    evidence_kind = str(artifact.get("evidence_kind") or reward_source or "unknown")
    evidence_id = str(
        provenance.get("evidence_id") or trial_id or provenance.get("task_name") or "evidence"
    )
    metadata = _dict(artifact.get("info"))
    for key in ("reward_source", "evidence_kind", "truth_owner"):
        metadata.pop(key, None)
    return EvidenceRecord(
        evidence_id=evidence_id,
        task_id=_string(provenance.get("task_name")),
        trial_id=trial_id,
        source=reward_source,
        kind=evidence_kind,
        truth_owner=_string(artifact.get("truth_owner")),
        status=str(artifact.get("loo_status") or "not_computed"),
        metric_name="reward",
        value=_number(artifact.get("reward")),
        components=_float_map(artifact.get("components")),
        diagnostics=_dict(artifact.get("diagnostics")),
        provenance=provenance,
        metadata=metadata,
    )


_SUCCESS_STATUSES = {"accepted", "ok", "pass", "passed", "success"}
_REJECTED_STATUSES = {"critical", "error", "failed", "not_computed", "reject", "rejected"}


def _is_verifier_metric(metric: EvidenceMetric) -> bool:
    return metric.truth_owner == "verifier"


def _metric_priority(metric: EvidenceMetric) -> tuple[int, int, int, int, int]:
    return (
        int(metric.value is not None),
        int(_is_verifier_metric(metric)),
        int(metric.source != "semantic_validation"),
        int(metric.kind == "verifier"),
        int(metric.name == "reward"),
    )


def _metric_from_evidence_record(evidence: EvidenceRecord) -> EvidenceMetric:
    return EvidenceMetric(
        name=evidence.metric_name or evidence.kind,
        value=evidence.value,
        status=evidence.status,
        source=evidence.source,
        kind=evidence.kind,
        truth_owner=evidence.truth_owner,
        diagnostics=evidence.diagnostics,
        metadata={
            "evidence_id": evidence.evidence_id,
            "task_id": evidence.task_id,
            "trial_id": evidence.trial_id,
        },
    )


def _metric_from_validation(validation: Mapping[str, Any]) -> EvidenceMetric | None:
    metric_name = _string(validation.get("semantic_metric"))
    metric_value = _number(validation.get("semantic_score"))
    if metric_name is None and metric_value is None:
        return None

    return EvidenceMetric(
        name=metric_name or "semantic_score",
        value=metric_value,
        status=_string(validation.get("semantic_status")),
        source="semantic_validation",
        kind=_string(validation.get("evidence_kind")),
        truth_owner=_string(validation.get("truth_owner")),
        diagnostics=_dict(validation.get("semantic_details")),
        metadata={
            "validation_ok": bool(validation.get("ok", False)),
            "repaired": bool(validation.get("repaired", False)),
        },
    )


def _candidate_status(
    validation: Mapping[str, Any],
    metrics: Sequence[EvidenceMetric],
    *,
    has_candidate_material: bool,
) -> str:
    validation_ok = validation.get("ok")
    verifier_metric_statuses = {
        status
        for metric in metrics
        if _is_verifier_metric(metric)
        if (status := _string(metric.status)) is not None
    }
    if validation_ok is False or verifier_metric_statuses & _REJECTED_STATUSES:
        return "rejected"
    if validation_ok is True or verifier_metric_statuses & _SUCCESS_STATUSES:
        return "ready_for_review"
    if has_candidate_material:
        return "needs_verification"
    return "incomplete"


def _primary_metric(metrics: Sequence[EvidenceMetric]) -> EvidenceMetric | None:
    if not metrics:
        return None
    return max(metrics, key=_metric_priority)


def _risks_from_validation_and_metrics(
    validation: Mapping[str, Any],
    metrics: Sequence[EvidenceMetric],
) -> list[str]:
    risks: list[str] = []
    if bool(validation.get("repaired")):
        risks.append("candidate required automated repair")
    if _string(validation.get("sample_data_error")) is not None:
        risks.append("validation fell back to degraded sample-data handling")

    semantic_status = _string(validation.get("semantic_status"))
    if semantic_status == "flag":
        risks.append("semantic validation flagged the candidate for operator review")
    elif semantic_status == "critical":
        risks.append("semantic validation marked the candidate as critical")

    semantic_details = _dict(validation.get("semantic_details"))
    if _string(semantic_details.get("hidden_data_error")) is not None:
        risks.append("holdout scoring reported hidden-data access issues")

    for metric in metrics:
        if metric.truth_owner == "heuristic":
            risks.append("heuristic-only evidence is present alongside verifier signals")
            break

    return risks


def _recommended_action(candidate_status: str) -> str:
    if candidate_status == "ready_for_review":
        return "review"
    if candidate_status == "rejected":
        return "repair_or_discard"
    if candidate_status == "needs_verification":
        return "run_verifier"
    return "regenerate"


def _summary_for_card(
    candidate_status: str,
    runtime: RuntimeDescriptor | None,
    primary_metric: EvidenceMetric | None,
    risks: Sequence[str],
) -> str:
    status_text = {
        "ready_for_review": "Candidate is ready for operator review",
        "rejected": "Candidate is rejected and should not be promoted",
        "needs_verification": "Candidate still needs verification",
        "incomplete": "Candidate generation is incomplete",
    }.get(candidate_status, "Candidate requires review")
    parts = [status_text]
    if runtime is not None:
        parts.append(f"runtime={runtime.name}")
    if primary_metric is not None and primary_metric.value is not None:
        parts.append(f"{primary_metric.name}={primary_metric.value:.4g}")
    if risks:
        parts.append(f"risks={len(risks)}")
    return "; ".join(parts)


def _runtime_descriptor(
    runtime: RuntimeDescriptor | Mapping[str, Any] | Any | None,
) -> RuntimeDescriptor | None:
    if runtime is None:
        return None
    if isinstance(runtime, RuntimeDescriptor):
        return runtime
    if isinstance(runtime, Mapping):
        payload = dict(runtime)
    else:
        payload = {
            "name": getattr(runtime, "name", None),
            "version": getattr(runtime, "version", None),
            "execution_backend": getattr(runtime, "execution_backend", None),
            "metadata": getattr(runtime, "metadata", {}),
        }
    if payload.get("name") is None:
        return None
    return RuntimeDescriptor.model_validate(payload)


def evidence_card_from_records(
    *,
    generation: GenerationRecord | None = None,
    trial: TrialRecord | None = None,
    trace: TraceRecord | None = None,
    evidence_records: Sequence[EvidenceRecord] = (),
) -> EvidenceCard:
    validation = {}
    if generation is not None:
        validation = _dict(generation.validation)
    if not validation and trial is not None:
        validation = _dict(trial.artifacts.get("validation"))

    metrics = [_metric_from_evidence_record(record) for record in evidence_records]
    validation_metric = _metric_from_validation(validation)
    if validation_metric is not None:
        metrics.append(validation_metric)

    primary_metric = _primary_metric(metrics)
    supporting_metrics = [metric for metric in metrics if metric is not primary_metric]

    runtime = None
    for candidate in (generation, trial):
        if candidate is not None and candidate.runtime is not None:
            runtime = _runtime_descriptor(candidate.runtime)
            break

    task_id = None
    generation_id = generation.record_id if generation is not None else None
    trace_id = None
    trial_id = None
    for value in (
        generation.task_id if generation is not None else None,
        trial.task_id if trial is not None else None,
        trace.task_id if trace is not None else None,
    ):
        if value is not None:
            task_id = value
            break
    for value in (
        trial.trial_id if trial is not None else None,
        trace.trial_id if trace is not None else None,
    ):
        if value is not None:
            trial_id = value
            break
    for value in (
        trace.trace_id if trace is not None else None,
        trial.trace_id if trial is not None else None,
    ):
        if value is not None:
            trace_id = value
            break

    has_candidate_material = any(
        value is not None and value != ""
        for value in (
            generation.raw_output if generation is not None else None,
            trial.program if trial is not None else None,
        )
    )
    candidate_status = _candidate_status(
        validation,
        metrics,
        has_candidate_material=has_candidate_material,
    )
    risks = _risks_from_validation_and_metrics(validation, metrics)
    recommended_action = _recommended_action(candidate_status)

    trial_artifacts = trial.artifacts if trial is not None else {}
    generation_metadata = generation.metadata if generation is not None else {}
    artifacts = _drop_none_values(
        {
            "prompt_id": _string(trial_artifacts.get("prompt_id"))
            or (generation.prompt_id if generation is not None else None),
            "split": _string(trial_artifacts.get("split"))
            or (generation.split if generation is not None else None),
            "env_name": _string(trial_artifacts.get("env_name"))
            or _string(generation_metadata.get("env_name")),
            "data_ref": _string(trial_artifacts.get("data_ref"))
            or _string(generation_metadata.get("data_ref")),
            "candidate_output_count": len(generation.candidate_outputs)
            if generation is not None
            else None,
            "trace_step_count": len(trace.steps) if trace is not None else None,
        }
    )

    base_id = trial_id or generation_id or trace_id or task_id or "candidate"
    return EvidenceCard(
        card_id=f"{base_id}:evidence-card",
        task_id=task_id,
        trial_id=trial_id,
        generation_id=generation_id,
        trace_id=trace_id,
        runtime=runtime,
        candidate_status=candidate_status,
        summary=_summary_for_card(candidate_status, runtime, primary_metric, risks),
        primary_metric=primary_metric,
        supporting_metrics=supporting_metrics,
        risks=risks,
        recommended_action=recommended_action,
        evidence_ids=[record.evidence_id for record in evidence_records],
        artifacts=artifacts,
        metadata={
            "source": "synthstats.platform_adapters.evidence_card",
            "validation_ok": bool(validation.get("ok", False)),
            "truth_owners": sorted(
                {owner for owner in (metric.truth_owner for metric in metrics) if owner is not None}
            ),
        },
    )


__all__ = [
    "evidence_card_from_records",
    "evidence_record_from_reward_artifact",
    "generation_record_from_dataset_row",
    "runtime_descriptor_from_dataset_row",
    "trace_record_from_dataset_row",
    "trace_record_from_messages",
    "trial_record_from_agent_result",
    "trial_record_from_dataset_row",
]
