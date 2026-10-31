"""Tests for Harbor InstalledAgent module."""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest


class TestHarborAgentModuleImport:
    def test_module_importable(self) -> None:
        import synthstats.integrations.harbor.agent as agent_mod

        assert agent_mod is not None

    def test_exports_available_flag(self) -> None:
        from synthstats.integrations.harbor.agent import HARBOR_AGENT_AVAILABLE

        assert isinstance(HARBOR_AGENT_AVAILABLE, bool)

    def test_exports_agent_class_name(self) -> None:
        import synthstats.integrations.harbor.agent as agent_mod

        assert hasattr(agent_mod, "SynthStatsInstalledAgent")

    def test_all_list_present(self) -> None:
        from synthstats.integrations.harbor.agent import __all__

        assert "SynthStatsInstalledAgent" in __all__
        assert "HARBOR_AGENT_AVAILABLE" in __all__


class TestHarborAgentImportGuard:
    def test_agent_none_without_harbor_sdk(self) -> None:
        """Without Harbor SDK installed, agent class should be None."""
        # we can't import Harbor in test env, so HARBOR_AGENT_AVAILABLE must be False
        # unless the SDK is actually present
        from synthstats.integrations.harbor.agent import (
            HARBOR_AGENT_AVAILABLE,
            SynthStatsInstalledAgent,
        )

        if not HARBOR_AGENT_AVAILABLE:
            assert SynthStatsInstalledAgent is None
        else:
            # SDK is installed: class must be a proper class
            assert isinstance(SynthStatsInstalledAgent, type)

    def test_make_agent_class_returns_none_without_sdk(self) -> None:
        """_make_agent_class() returns None when BaseInstalledAgent unavailable."""
        import synthstats.integrations.harbor.agent as agent_mod

        orig_base = agent_mod._BaseInstalledAgent
        try:
            agent_mod._BaseInstalledAgent = None
            result = agent_mod._make_agent_class()
            assert result is None
        finally:
            agent_mod._BaseInstalledAgent = orig_base


class TestHarborAgentFactoryFunction:
    def test_make_agent_class_returns_class_with_sdk(self) -> None:
        """If _BaseInstalledAgent is set, factory produces a class."""
        import synthstats.integrations.harbor.agent as agent_mod

        if agent_mod._BaseInstalledAgent is None:
            pytest.skip("Harbor SDK not installed")

        cls = agent_mod._make_agent_class()
        assert cls is not None
        assert isinstance(cls, type)

    def test_make_agent_class_sets_available_flag(self) -> None:
        """HARBOR_AGENT_AVAILABLE reflects whether the factory produced a class."""
        import synthstats.integrations.harbor.agent as agent_mod

        expected = agent_mod.SynthStatsInstalledAgent is not None
        assert agent_mod.HARBOR_AGENT_AVAILABLE is expected

    def test_make_agent_class_implements_current_harbor_contract(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        import synthstats.integrations.harbor.agent as agent_mod

        class _FakeInstalledAgentBase:
            def __init__(self, **kwargs):
                self.logs_dir = kwargs["logs_dir"]

        monkeypatch.setenv("LLM_API_BASE", "http://example.test/v1")
        orig_base = agent_mod._BaseInstalledAgent
        try:
            agent_mod._BaseInstalledAgent = _FakeInstalledAgentBase
            cls = agent_mod._make_agent_class()
        finally:
            agent_mod._BaseInstalledAgent = orig_base

        assert cls is not None
        assert cls.name() == "synthstats"

        agent = cls(logs_dir=tmp_path / "agent")

        result_path = agent.logs_dir / "synthstats_result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps({"reward": 1.25, "program": "x = 1"}))

        class _Context:
            metadata = None

        context = _Context()
        agent.populate_context_post_run(context)

        assert context.metadata == {"reward": 1.25, "program": "x = 1"}

    def test_agent_install_and_run_execute_inside_harbor_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        from types import SimpleNamespace

        import synthstats.integrations.harbor.agent as agent_mod

        calls: list[tuple[str, str, dict[str, str] | None]] = []

        class _FakeInstalledAgentBase:
            def __init__(self, **kwargs):
                self.logs_dir = kwargs["logs_dir"]
                self._extra_env = kwargs.get("env", {})

            async def exec_as_root(self, environment, *, command: str, env=None):
                calls.append(("root", command, env))
                return SimpleNamespace(return_code=0, stdout="", stderr="")

            async def exec_as_agent(self, environment, *, command: str, env=None):
                calls.append(("agent", command, env))
                return SimpleNamespace(return_code=0, stdout="", stderr="")

        orig_base = agent_mod._BaseInstalledAgent
        try:
            agent_mod._BaseInstalledAgent = _FakeInstalledAgentBase
            cls = agent_mod._make_agent_class()
        finally:
            agent_mod._BaseInstalledAgent = orig_base

        assert cls is not None
        runtime_env = {
            "LLM_API_BASE": "http://example.test/v1",
            "TASK_CONFIG": json.dumps({"env": {"name": "dugongs"}}),
            "TASK_PROVENANCE": json.dumps({"task_name": "dugongs"}),
            "MODEL_NAME": "Qwen/Qwen3.5-0.8B",
            "SAMPLING_PARAMS": json.dumps({"max_generate_length": 64, "temperature": 0.7}),
            "MAX_SEQUENCE_LENGTH": "512",
            "MAX_TURNS": "4",
        }
        agent = cls(logs_dir=tmp_path / "agent", env=runtime_env)
        fake_env = object()
        asyncio.run(agent.install(fake_env))
        asyncio.run(agent.run("Fit dugongs.", fake_env, SimpleNamespace(metadata=None)))

        assert calls[0][0] == "root"
        assert "/logs/agent" in calls[0][1]
        assert calls[1] == ("agent", "python --version", None)
        assert calls[2][0] == "agent"
        assert calls[2][1] == "python -m synthstats.integrations.harbor.episode"
        assert calls[2][2] is not None
        assert calls[2][2]["TASK_INSTRUCTION"] == "Fit dugongs."
        assert calls[2][2]["MODEL_NAME"] == "Qwen/Qwen3.5-0.8B"
        assert "/workspace/synthstats" in calls[2][2].get("PYTHONPATH", "")
        assert json.loads(calls[2][2]["TASK_PROVENANCE"]) == {"task_name": "dugongs"}


class TestHarborAgentLogging:
    def test_write_agent_result_log_writes_harbor_verifier_payload(self, tmp_path) -> None:
        from synthstats.integrations.harbor.agent import write_agent_result_log

        output_path = tmp_path / "logs" / "agent" / "synthstats_result.json"
        write_agent_result_log(
            {
                "program": "print('hi')",
                "reward": 1.5,
                "turns": 3,
                "task_config": {"env": {"name": "dugongs"}},
                "task_provenance": {"source": "local_task_package"},
            },
            path=output_path,
        )

        payload = json.loads(output_path.read_text())
        assert payload["program"] == "print('hi')"
        assert payload["reward"] == pytest.approx(1.5)
        assert payload["turns"] == 3
        assert payload["language"] == "pymc"
        assert payload["task_config"] == {"env": {"name": "dugongs"}}
        assert payload["task_provenance"] == {"source": "local_task_package"}
        assert payload["reward_info"]["reward_source"] == "unknown"
        assert payload["reward_info"]["evidence_kind"] == "heuristic"
        assert payload["reward_info"]["truth_owner"] == "heuristic"

    def test_write_agent_result_log_annotates_validation_reward_info(self, tmp_path) -> None:
        from synthstats.integrations.harbor.agent import write_agent_result_log

        output_path = tmp_path / "logs" / "agent" / "synthstats_result.json"
        write_agent_result_log(
            {
                "program": "print('hi')",
                "reward": -0.123,
                "reward_info": {
                    "source": "validation.semantic_score",
                    "semantic_score": -0.123,
                },
            },
            path=output_path,
        )

        payload = json.loads(output_path.read_text())
        assert payload["reward_info"] == {
            "source": "validation.semantic_score",
            "reward_source": "validation.semantic_score",
            "evidence_kind": "verifier",
            "truth_owner": "verifier",
            "semantic_score": -0.123,
        }

    def test_require_llm_api_base_fails_without_explicit_value(self) -> None:
        from synthstats.integrations.harbor.agent import require_llm_api_base

        with pytest.raises(ValueError, match="LLM_API_BASE"):
            require_llm_api_base({})


class TestNormalizeAgentRewardInfo:
    def test_none_returns_annotated(self) -> None:
        from synthstats.integrations.harbor.agent import normalize_agent_reward_info

        result = normalize_agent_reward_info(None)
        assert result["reward_source"] == "unknown"
        assert result["evidence_kind"] == "heuristic"
        assert result["truth_owner"] == "heuristic"

    def test_empty_dict_returns_annotated(self) -> None:
        from synthstats.integrations.harbor.agent import normalize_agent_reward_info

        result = normalize_agent_reward_info({})
        assert result["reward_source"] == "unknown"
        assert result["evidence_kind"] == "heuristic"
        assert result["truth_owner"] == "heuristic"

    def test_custom_payload_preserves_and_annotates(self) -> None:
        from synthstats.integrations.harbor.agent import normalize_agent_reward_info

        result = normalize_agent_reward_info({"custom": 42})
        assert result["custom"] == 42
        assert result["reward_source"] == "unknown"
        assert result["evidence_kind"] == "heuristic"
        assert result["truth_owner"] == "heuristic"

    def test_partial_provenance_fills_remaining(self) -> None:
        from synthstats.integrations.harbor.agent import normalize_agent_reward_info

        result = normalize_agent_reward_info({"source": "validation.elpd"})
        assert result["reward_source"] == "validation.elpd"
        assert result["evidence_kind"] == "verifier"
        assert result["truth_owner"] == "verifier"

    def test_fully_annotated_passthrough(self) -> None:
        from synthstats.integrations.harbor.agent import normalize_agent_reward_info

        full = {
            "source": "verifier",
            "reward_source": "verifier",
            "evidence_kind": "verifier",
            "truth_owner": "verifier",
            "extra": 99,
        }
        result = normalize_agent_reward_info(full)
        assert result == full


class TestHarborAgentPrompting:
    def test_build_action_spec_uses_codec_contract(self) -> None:
        from synthstats.integrations.harbor.agent import build_action_spec

        class _FakeTask:
            def tool_schemas(self) -> list[dict]:
                return [
                    {
                        "function": {
                            "name": "query",
                            "description": "collect data",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    }
                ]

        class _FakeCodec:
            def format_action_spec(self, tools) -> str:
                assert len(tools) == 1
                assert tools[0].name == "query"
                return "ACTION SPEC"

        class _FakeEnv:
            task = _FakeTask()
            codec = _FakeCodec()

        action_spec = build_action_spec(_FakeEnv())

        assert "ACTION SPEC" in action_spec
        assert "exactly one action" in action_spec

    def test_prime_chat_history_injects_instruction_and_action_spec(self) -> None:
        from synthstats.integrations.harbor.agent import prime_chat_history

        chat_history = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "Observations:\nNo observations yet."},
        ]

        primed = prime_chat_history(
            chat_history,
            instruction="Solve the task.",
            action_spec="USE TOOL TAGS",
        )

        assert primed[0]["role"] == "system"
        assert "system prompt" in primed[0]["content"]
        assert "USE TOOL TAGS" in primed[0]["content"]
        assert primed[1] == {"role": "user", "content": "Solve the task."}
        assert primed[2]["content"].startswith("Observations:")

    def test_render_openai_message_serializes_tool_call(self) -> None:
        from synthstats.integrations.harbor.agent import render_openai_message
        from synthstats.runtime.codecs import NativeToolCallingCodec

        rendered = render_openai_message(
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "submit",
                            "arguments": json.dumps(
                                {"code": "import pymc as pm\nwith pm.Model():\n    pass"}
                            ),
                        },
                    }
                ],
            }
        )

        action = NativeToolCallingCodec().parse(rendered)
        assert action.language == "pymc"
        assert "pm.Model" in action.code

    def test_call_llm_sends_structured_messages_and_tools(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        import synthstats.integrations.harbor.agent as agent_mod

        captured: dict[str, object] = {}

        class _FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_query",
                                        "type": "function",
                                        "function": {
                                            "name": "query",
                                            "arguments": json.dumps({"query": "school=1"}),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }

        class _FakeClient:
            async def post(self, path: str, json: dict, timeout: float) -> _FakeResponse:
                captured["path"] = path
                captured["json"] = json
                captured["timeout"] = timeout
                return _FakeResponse()

        class _FakeInstalledAgentBase:
            def __init__(self, **kwargs):
                self.logs_dir = kwargs["logs_dir"]

        monkeypatch.setenv("LLM_API_BASE", "http://example.test/v1")
        orig_base = agent_mod._BaseInstalledAgent
        try:
            agent_mod._BaseInstalledAgent = _FakeInstalledAgentBase
            cls = agent_mod._make_agent_class()
        finally:
            agent_mod._BaseInstalledAgent = orig_base

        assert cls is not None
        agent = cls(logs_dir=tmp_path / "agent")
        messages = [{"role": "system", "content": "prompt"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "query",
                    "description": "collect data",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        rendered = asyncio.run(agent._call_llm(_FakeClient(), messages, tool_schemas=tools))

        request_json = cast(dict[str, Any], captured["json"])

        assert captured["path"] == "/chat/completions"
        assert request_json["messages"] == messages
        assert request_json["tools"] == tools
        assert request_json["tool_choice"] == "required"
        assert request_json["parallel_tool_calls"] is False
        assert "query" in rendered
