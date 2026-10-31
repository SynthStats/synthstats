"""Canonical reward provenance semantics for Harbor integrations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from synthstats.core.types import Reward


class EvidenceKind(StrEnum):
    VERIFIER = "verifier"
    HEURISTIC = "heuristic"
    AUDIT = "audit"


class TruthOwner(StrEnum):
    VERIFIER = "verifier"
    HEURISTIC = "heuristic"
    AUDITOR = "auditor"


@dataclass(frozen=True)
class RewardSemantics:
    reward_source: str
    evidence_kind: EvidenceKind
    truth_owner: TruthOwner


_VERIFIER_REWARD_SOURCES = frozenset(
    {
        "verifier",
        "verifier_result",
        "verifier_missing_program",
    }
)
_VERIFIER_REWARD_SOURCE_PREFIXES = ("validation.",)


def _is_verifier_reward_source(normalized: str) -> bool:
    return normalized in _VERIFIER_REWARD_SOURCES or normalized.startswith(
        _VERIFIER_REWARD_SOURCE_PREFIXES
    )


def reward_semantics_for_source(reward_source: str | None) -> RewardSemantics:
    """Map a reward provenance label onto the canonical truth taxonomy."""
    normalized = str(reward_source or "unknown").strip() or "unknown"
    if _is_verifier_reward_source(normalized):
        return RewardSemantics(
            reward_source=normalized,
            evidence_kind=EvidenceKind.VERIFIER,
            truth_owner=TruthOwner.VERIFIER,
        )

    return RewardSemantics(
        reward_source=normalized,
        evidence_kind=EvidenceKind.HEURISTIC,
        truth_owner=TruthOwner.HEURISTIC,
    )


def annotate_reward_info(
    info: Mapping[str, Any] | None = None,
    *,
    reward_source: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Fill in explicit reward provenance semantics without clobbering payloads."""
    annotated = dict(info or {})
    semantics = reward_semantics_for_source(
        reward_source or annotated.get("reward_source") or annotated.get("source") or source
    )

    if source is not None:
        annotated.setdefault("source", source)
    else:
        annotated.setdefault("source", semantics.reward_source)

    annotated.setdefault("reward_source", semantics.reward_source)
    annotated.setdefault("evidence_kind", semantics.evidence_kind.value)
    annotated.setdefault("truth_owner", semantics.truth_owner.value)
    return annotated


def reward_info_is_verifier_owned(info: Mapping[str, Any] | None) -> bool:
    """Return whether reward metadata is explicitly verifier-owned."""
    annotated = annotate_reward_info(info)
    return (
        annotated.get("evidence_kind") == EvidenceKind.VERIFIER.value
        and annotated.get("truth_owner") == TruthOwner.VERIFIER.value
    )


def reward_is_verifier_owned(reward: Reward) -> bool:
    """Return whether a Reward object carries verifier-owned semantics."""
    return reward_info_is_verifier_owned(reward.info)
