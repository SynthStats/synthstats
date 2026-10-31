from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from synthstats.integrations.skyrl.runner import resolve_harbor_generator_class
from synthstats.train.runners.harbor import (
    HARBOR_GENERATOR_CONFIG_KEYS,
    extract_harbor_generator_kwargs,
)


class TestRunnerHarbor:
    def test_extract_harbor_generator_kwargs_from_mapping(self):
        kwargs = extract_harbor_generator_kwargs(
            {
                "agent_import_path": "pkg.agent:Agent",
                "agent_model_name": "Qwen/Qwen3.5-4B",
                "llm_api_base": "http://127.0.0.1:8000/v1",
                "collect_rollout_details": True,
                "unused": "ignored",
            }
        )

        assert kwargs == {
            "agent_import_path": "pkg.agent:Agent",
            "agent_model_name": "Qwen/Qwen3.5-4B",
            "llm_api_base": "http://127.0.0.1:8000/v1",
            "collect_rollout_details": True,
        }

    def test_extract_harbor_generator_kwargs_from_object(self):
        kwargs = extract_harbor_generator_kwargs(
            SimpleNamespace(
                sandbox_provider="docker",
                inference_port=8000,
                agent_timeout_sec=120.0,
            )
        )

        assert kwargs == {
            "sandbox_provider": "docker",
            "inference_port": 8000,
            "agent_timeout_sec": 120.0,
        }

    def test_exported_key_order_matches_supported_projection_surface(self):
        assert HARBOR_GENERATOR_CONFIG_KEYS == (
            "agent_import_path",
            "agent_model_name",
            "sandbox_provider",
            "inference_port",
            "llm_api_base",
            "agent_timeout_sec",
            "collect_rollout_details",
        )

    def test_resolve_harbor_generator_class_returns_none_for_non_harbor(self):
        exp_cfg = SimpleNamespace(generator=SimpleNamespace(type="other"))

        assert resolve_harbor_generator_class(exp_cfg) is None

    def test_resolve_harbor_generator_class_returns_none_without_generator(self):
        assert resolve_harbor_generator_class(SimpleNamespace()) is None

    def test_resolve_harbor_generator_class_raises_when_harbor_missing(self):
        with patch.dict(
            "sys.modules",
            {
                "synthstats.integrations.harbor.generator": SimpleNamespace(
                    HARBOR_GENERATOR_AVAILABLE=False,
                    SynthStatsHarborGenerator=object,
                )
            },
        ):
            with pytest.raises(ImportError, match="generator.type=harbor"):
                resolve_harbor_generator_class(
                    SimpleNamespace(generator=SimpleNamespace(type="harbor"))
                )

    def test_resolve_harbor_generator_class_returns_generator_when_available(self):
        fake_generator = type("FakeHarborGenerator", (), {})

        with patch.dict(
            "sys.modules",
            {
                "synthstats.integrations.harbor.generator": SimpleNamespace(
                    HARBOR_GENERATOR_AVAILABLE=True,
                    SynthStatsHarborGenerator=fake_generator,
                )
            },
        ):
            assert (
                resolve_harbor_generator_class(
                    SimpleNamespace(generator=SimpleNamespace(type="harbor"))
                )
                is fake_generator
            )
