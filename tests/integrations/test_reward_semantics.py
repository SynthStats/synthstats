"""Tests for canonical reward provenance semantics."""

from __future__ import annotations

from synthstats.core.reward_semantics import (
    EvidenceKind,
    TruthOwner,
    annotate_reward_info,
    reward_info_is_verifier_owned,
    reward_semantics_for_source,
)


class TestAnnotateRewardInfo:
    def test_empty_dict_produces_all_provenance_keys(self) -> None:
        result = annotate_reward_info({})
        assert "reward_source" in result
        assert "evidence_kind" in result
        assert "truth_owner" in result

    def test_none_input_produces_all_provenance_keys(self) -> None:
        result = annotate_reward_info(None)
        assert result["reward_source"] == "unknown"
        assert result["evidence_kind"] == "heuristic"
        assert result["truth_owner"] == "heuristic"

    def test_idempotent(self) -> None:
        original = {"custom": 42}
        once = annotate_reward_info(original)
        twice = annotate_reward_info(once)
        assert once == twice

    def test_preserves_existing_custom_keys(self) -> None:
        result = annotate_reward_info({"custom": 42, "extra": "data"})
        assert result["custom"] == 42
        assert result["extra"] == "data"
        assert "reward_source" in result

    def test_does_not_clobber_existing_reward_source(self) -> None:
        result = annotate_reward_info({"reward_source": "verifier"})
        assert result["reward_source"] == "verifier"
        assert result["evidence_kind"] == "verifier"
        assert result["truth_owner"] == "verifier"

    def test_does_not_clobber_existing_evidence_kind(self) -> None:
        result = annotate_reward_info(
            {"reward_source": "custom_judge", "evidence_kind": "verifier"}
        )
        assert result["evidence_kind"] == "verifier"

    def test_source_kwarg_sets_source_field(self) -> None:
        result = annotate_reward_info({}, source="validation.elpd")
        assert result["source"] == "validation.elpd"
        assert result["evidence_kind"] == "verifier"

    def test_reward_source_kwarg_takes_precedence(self) -> None:
        result = annotate_reward_info(
            {"source": "heuristic_judge"},
            reward_source="verifier",
        )
        assert result["reward_source"] == "verifier"
        assert result["evidence_kind"] == "verifier"


class TestRewardSemanticsForSource:
    def test_verifier_source(self) -> None:
        sem = reward_semantics_for_source("verifier")
        assert sem.evidence_kind == EvidenceKind.VERIFIER
        assert sem.truth_owner == TruthOwner.VERIFIER

    def test_verifier_result_source(self) -> None:
        sem = reward_semantics_for_source("verifier_result")
        assert sem.evidence_kind == EvidenceKind.VERIFIER

    def test_verifier_missing_program_source(self) -> None:
        sem = reward_semantics_for_source("verifier_missing_program")
        assert sem.evidence_kind == EvidenceKind.VERIFIER

    def test_validation_prefix_is_verifier(self) -> None:
        sem = reward_semantics_for_source("validation.elpd")
        assert sem.evidence_kind == EvidenceKind.VERIFIER
        assert sem.truth_owner == TruthOwner.VERIFIER

    def test_unknown_source_is_heuristic(self) -> None:
        sem = reward_semantics_for_source("unknown")
        assert sem.evidence_kind == EvidenceKind.HEURISTIC
        assert sem.truth_owner == TruthOwner.HEURISTIC

    def test_none_source_is_heuristic(self) -> None:
        sem = reward_semantics_for_source(None)
        assert sem.evidence_kind == EvidenceKind.HEURISTIC

    def test_arbitrary_source_is_heuristic(self) -> None:
        sem = reward_semantics_for_source("llm_judge")
        assert sem.evidence_kind == EvidenceKind.HEURISTIC


class TestRewardInfoIsVerifierOwned:
    def test_verifier_info_returns_true(self) -> None:
        info = annotate_reward_info({}, reward_source="verifier")
        assert reward_info_is_verifier_owned(info) is True

    def test_heuristic_info_returns_false(self) -> None:
        info = annotate_reward_info({}, reward_source="llm_judge")
        assert reward_info_is_verifier_owned(info) is False

    def test_none_info_returns_false(self) -> None:
        assert reward_info_is_verifier_owned(None) is False

    def test_empty_dict_returns_false(self) -> None:
        assert reward_info_is_verifier_owned({}) is False
