"""A critic infrastructure failure must never become a training reward."""

import sys
from types import SimpleNamespace

import pytest

from synthstats.core.types import Trajectory
from synthstats.judges.composite import CompositeJudge
from synthstats.judges.llm_critic import LLMCriticConfig, LLMCriticJudge, LLMCriticUnavailableError


def score(judge):
    return judge.score(
        task_name="dugongs",
        trajectory=Trajectory.empty_stub([]),
        artifacts={"program": "x = 1"},
    )


def test_missing_litellm_fails_instead_of_scoring(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", None)

    with pytest.raises(LLMCriticUnavailableError, match="LiteLLM"):
        score(LLMCriticJudge())


def test_missing_configured_credential_fails_instead_of_scoring(monkeypatch):
    def forbidden_request(**kwargs):
        raise AssertionError("A missing credential must prevent a provider request")

    monkeypatch.delenv("SYNTHSTATS_TEST_CRITIC_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=forbidden_request))
    judge = LLMCriticJudge(LLMCriticConfig(api_key_env_var="SYNTHSTATS_TEST_CRITIC_KEY"))

    with pytest.raises(LLMCriticUnavailableError, match="SYNTHSTATS_TEST_CRITIC_KEY"):
        score(judge)


def test_failed_request_fails_without_exposing_provider_details(monkeypatch):
    def unavailable(**kwargs):
        raise OSError("private provider details")

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=unavailable))

    with pytest.raises(LLMCriticUnavailableError, match="LLM critic request failed") as error:
        score(LLMCriticJudge())
    assert "private provider details" not in str(error.value)


def test_empty_provider_response_fails_instead_of_scoring(monkeypatch):
    def empty_response(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=empty_response))

    with pytest.raises(LLMCriticUnavailableError, match="LLM critic.*no critique"):
        score(LLMCriticJudge())


def test_composite_does_not_convert_an_unavailable_critic_into_a_reward(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", None)
    judge = CompositeJudge([(LLMCriticJudge(), 1.0)])

    with pytest.raises(LLMCriticUnavailableError, match="LiteLLM"):
        score(judge)
