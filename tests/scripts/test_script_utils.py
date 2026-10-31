from __future__ import annotations

import logging

from conftest import load_script_module as _load_script_module


def test_importing_scripts_does_not_configure_logging(monkeypatch):
    calls: list[tuple[tuple, dict]] = []

    def fake_basic_config(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)

    _load_script_module("compute_atif_rewards")
    _load_script_module("compute_pilot_elpd")
    _load_script_module("sft_to_atif")
    _load_script_module("generate_pilot_programs")

    assert calls == []
