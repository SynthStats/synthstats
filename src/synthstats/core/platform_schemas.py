from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

RECORD_SCHEMA_VERSION = "ppl.platform.record.v0"
CARD_SCHEMA_VERSION = "ppl.platform.card.v0"
REPORT_SCHEMA_VERSION = "ppl.platform.report.v0"


class RuntimeDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str | None = None
    execution_backend: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = RECORD_SCHEMA_VERSION
    view_id: str
    task_id: str | None = None
    split: str | None = None
    content: str
    format: str = "text/plain"
    provenance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = RECORD_SCHEMA_VERSION
    record_id: str
    task_id: str | None = None
    prompt_id: str | None = None
    split: str | None = None
    runtime: RuntimeDescriptor | None = None
    prompt: PromptView | None = None
    raw_output: str
    candidate_outputs: list[str] = Field(default_factory=list)
    provider_response: dict[str, Any] = Field(default_factory=dict)
    parsed_program: str | None = None
    validation: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    role: str
    content: str
    tool_call_id: str | None = None
    reasoning_content: str | None = None
    evidence_kind: str | None = None
    truth_owner: str | None = None
    semantic_status: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = RECORD_SCHEMA_VERSION
    trace_id: str
    task_id: str | None = None
    trial_id: str | None = None
    steps: list[TraceStep] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrialRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = RECORD_SCHEMA_VERSION
    trial_id: str
    task_id: str | None = None
    language: str | None = None
    program: str | None = None
    reward: float | None = None
    runtime: RuntimeDescriptor | None = None
    trace_id: str | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = RECORD_SCHEMA_VERSION
    evidence_id: str
    task_id: str | None = None
    trial_id: str | None = None
    source: str
    kind: str
    truth_owner: str | None = None
    status: str
    metric_name: str | None = None
    value: float | None = None
    components: dict[str, float] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: float | None = None
    status: str | None = None
    source: str | None = None
    kind: str | None = None
    truth_owner: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = CARD_SCHEMA_VERSION
    card_id: str
    task_id: str | None = None
    trial_id: str | None = None
    generation_id: str | None = None
    trace_id: str | None = None
    runtime: RuntimeDescriptor | None = None
    candidate_status: str
    operator_status: str = "pending_review"
    summary: str
    primary_metric: EvidenceMetric | None = None
    supporting_metrics: list[EvidenceMetric] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommended_action: str = "review"
    evidence_ids: list[str] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "CARD_SCHEMA_VERSION",
    "EvidenceCard",
    "EvidenceMetric",
    "EvidenceRecord",
    "GenerationRecord",
    "PromptView",
    "RECORD_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "RuntimeDescriptor",
    "TraceRecord",
    "TraceStep",
    "TrialRecord",
]
