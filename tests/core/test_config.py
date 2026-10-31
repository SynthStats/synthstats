"""Tests for core/config.py utilities."""

from __future__ import annotations

from synthstats.core.config import sanitize_config


class TestSanitizeConfig:
    def test_redacts_api_key(self) -> None:
        cfg = {"model": "qwen", "api_key": "fake-key-1"}
        result = sanitize_config(cfg)
        assert result["model"] == "qwen"
        assert result["api_key"] == "***REDACTED***"

    def test_redacts_nested_api_key(self) -> None:
        cfg = {"policy": {"model": "qwen", "api_key": "fake-key-2"}, "seed": 42}
        result = sanitize_config(cfg)
        assert result["policy"]["api_key"] == "***REDACTED***"
        assert result["policy"]["model"] == "qwen"
        assert result["seed"] == 42

    def test_redacts_deeply_nested(self) -> None:
        cfg = {"a": {"b": {"c": {"api_key": "fake-key-3"}}}}
        result = sanitize_config(cfg)
        assert result["a"]["b"]["c"]["api_key"] == "***REDACTED***"

    def test_redacts_password_and_credential(self) -> None:
        cfg = {"db_password": "fake-pass-1", "user_credential": "fake-cred-1", "name": "ok"}
        result = sanitize_config(cfg)
        assert result["db_password"] == "***REDACTED***"
        assert result["user_credential"] == "***REDACTED***"
        assert result["name"] == "ok"

    def test_preserves_non_secret_fields(self) -> None:
        cfg = {"model": "qwen", "lr": 0.001, "steps": 1000}
        result = sanitize_config(cfg)
        assert result == cfg

    def test_handles_list_of_dicts(self) -> None:
        cfg = {"items": [{"api_key": "fake-key-3", "v": 1}, {"v": 2}]}
        result = sanitize_config(cfg)
        assert result["items"][0]["api_key"] == "***REDACTED***"
        assert result["items"][0]["v"] == 1
        assert result["items"][1]["v"] == 2

    def test_does_not_mutate_original(self) -> None:
        cfg = {"api_key": "original"}
        result = sanitize_config(cfg)
        assert cfg["api_key"] == "original"
        assert result["api_key"] == "***REDACTED***"

    def test_empty_dict(self) -> None:
        assert sanitize_config({}) == {}
