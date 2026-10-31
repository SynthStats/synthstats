from __future__ import annotations

import json
import types

import pytest
from conftest import load_script_module


def test_extract_haskell_code_falls_back_to_module_block():
    script = load_script_module("generate_pilot_programs")
    text = "Some preamble\nmodule Submission where\nimport LazyPPL\nanswer = 1\n"

    code = script.extract_haskell_code(text)

    assert code == "module Submission where\nimport LazyPPL\nanswer = 1"


def test_build_user_prompt_includes_schema_and_rows():
    script = load_script_module("generate_pilot_programs")
    task = {
        "description": "Model dugongs",
        "columns": [
            {"name": "age", "hs_type": "Double"},
            {"name": "length", "hs_type": "Double"},
        ],
        "observations": [
            {"age": 1.0, "length": 2.0},
            {"age": 3.0, "length": 4.0},
        ],
        "n_observations": 2,
    }

    prompt = script.build_user_prompt(task)

    assert "age :: Double, length :: Double" in prompt
    assert "age=1.0, length=2.0" in prompt
    assert "Number of observations: 2" in prompt


def test_build_atif_uses_shared_agent_metadata():
    script = load_script_module("generate_pilot_programs")
    utils = load_script_module("_utils")

    atif = script.build_atif(
        env_name="dugongs",
        idx=2,
        system_msg="system",
        user_msg="user",
        completion="module Submission where",
        code="module Submission where\nvalue = 1",
        token_ids=[1, 2],
        logprobs=[-0.1, -0.2],
        compiles=True,
        reward_source="validation.compilation",
        model_name="demo-model",
    )

    assert atif["schema_version"] == utils.ATIF_SCHEMA_VERSION
    assert atif["agent"] == utils.build_atif_agent("demo-model")
    assert atif["session_id"] == "pilot-dugongs-0002"
    assert atif["final_metrics"]["extra"]["reward"] == 1.0
    assert atif["final_metrics"]["extra"]["reward_source"] == "validation.compilation"
    assert atif["final_metrics"]["extra"]["evidence_kind"] == "verifier"
    assert atif["final_metrics"]["extra"]["truth_owner"] == "verifier"


def test_main_writes_atif_with_explicit_paths(tmp_path, monkeypatch):
    script = load_script_module("generate_pilot_programs")
    pilot_data_dir = tmp_path / "pilot_data"
    output_dir = tmp_path / "atif_trajectories"
    pilot_data_dir.mkdir()
    (pilot_data_dir / "lazyppl_pilot_dugongs.json").write_text(
        json.dumps(
            {
                "description": "Model dugongs",
                "columns": [
                    {"name": "age", "hs_type": "Double"},
                    {"name": "length", "hs_type": "Double"},
                ],
                "observations": [{"age": 1.0, "length": 2.0}],
                "n_observations": 1,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        script,
        "generate_programs_api",
        lambda *args, **kwargs: [
            {
                "text": "```haskell\nmodule Submission where\nimport LazyPPL\nvalue = 1\n```",
                "token_ids": [11, 12, 13],
                "logprobs": [-0.1, -0.2, -0.3],
            }
        ],
    )

    args = types.SimpleNamespace(
        env="dugongs",
        n=1,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=256,
        model="demo-model",
        no_typecheck=True,
        api=True,
        api_base="https://example.invalid/v1",
        api_key="fake-key-1",
        output_dir=None,
    )

    assert script.main(args=args, pilot_data_dir=pilot_data_dir, output_dir=output_dir) == 0

    payload = json.loads((output_dir / "pilot-dugongs-0000.json").read_text())

    assert payload["agent"]["model_name"] == "demo-model"
    assert payload["final_metrics"]["extra"]["reward"] == 1.0
    assert payload["final_metrics"]["extra"]["compiles"] is True
    assert payload["final_metrics"]["extra"]["reward_source"] == "heuristic.compilation_assumed"
    assert payload["final_metrics"]["extra"]["evidence_kind"] == "heuristic"
    assert payload["final_metrics"]["extra"]["truth_owner"] == "heuristic"
    assert payload["steps"][2]["tool_calls"][0]["arguments"]["code"] == (
        "module Submission where\nimport LazyPPL\nvalue = 1"
    )


def test_main_returns_error_for_unknown_env(tmp_path):
    script = load_script_module("generate_pilot_programs")
    args = types.SimpleNamespace(
        env="unknown_env",
        n=1,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=256,
        model="demo-model",
        no_typecheck=True,
        api=True,
        api_base="https://example.invalid/v1",
        api_key="fake-key-1",
        output_dir=None,
    )

    assert script.main(args=args, pilot_data_dir=tmp_path, output_dir=tmp_path / "out") == 1


def test_main_returns_error_when_model_load_fails(tmp_path, monkeypatch):
    script = load_script_module("generate_pilot_programs")
    pilot_data_dir = tmp_path / "pilot_data"
    pilot_data_dir.mkdir()
    (pilot_data_dir / "lazyppl_pilot_dugongs.json").write_text(
        json.dumps(
            {
                "description": "Model dugongs",
                "columns": [{"name": "age", "hs_type": "Double"}],
                "observations": [{"age": 1.0}],
                "n_observations": 1,
            }
        ),
        encoding="utf-8",
    )

    def raise_load_error(model_id: str):
        raise RuntimeError(f"cannot load {model_id}")

    monkeypatch.setattr(script, "load_model_and_tokenizer", raise_load_error)

    args = types.SimpleNamespace(
        env="dugongs",
        n=1,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=256,
        model="demo-model",
        no_typecheck=True,
        api=False,
        api_base=None,
        api_key=None,
        output_dir=None,
    )

    assert script.main(args=args, pilot_data_dir=pilot_data_dir, output_dir=tmp_path / "out") == 1


def test_main_records_failed_typecheck(tmp_path, monkeypatch):
    script = load_script_module("generate_pilot_programs")
    pilot_data_dir = tmp_path / "pilot_data"
    output_dir = tmp_path / "atif_trajectories"
    pilot_data_dir.mkdir()
    (pilot_data_dir / "lazyppl_pilot_dugongs.json").write_text(
        json.dumps(
            {
                "description": "Model dugongs",
                "columns": [{"name": "age", "hs_type": "Double"}],
                "observations": [{"age": 1.0}],
                "n_observations": 1,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        script,
        "generate_programs_api",
        lambda *args, **kwargs: [
            {
                "text": "```haskell\nmodule Submission where\nvalue = 1\n```",
                "token_ids": [1, 2],
                "logprobs": [-0.1, -0.2],
            }
        ],
    )
    monkeypatch.setattr(
        script,
        "typecheck",
        lambda code, module_name: (
            False,
            f"{module_name}: error: could not match expected type",
        ),
    )

    args = types.SimpleNamespace(
        env="dugongs",
        n=1,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=256,
        model="demo-model",
        no_typecheck=False,
        api=True,
        api_base="https://example.invalid/v1",
        api_key="fake-key-1",
        output_dir=None,
    )

    assert script.main(args=args, pilot_data_dir=pilot_data_dir, output_dir=output_dir) == 0

    payload = json.loads((output_dir / "pilot-dugongs-0000.json").read_text())
    assert payload["final_metrics"]["extra"]["reward"] == 1e-08
    assert payload["final_metrics"]["extra"]["compiles"] is False
    assert payload["final_metrics"]["extra"]["reward_source"] == "validation.compilation"
    assert payload["final_metrics"]["extra"]["evidence_kind"] == "verifier"
    assert payload["final_metrics"]["extra"]["truth_owner"] == "verifier"


def test_generate_programs_api_requires_api_key(monkeypatch):
    script = load_script_module("generate_pilot_programs")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="No API key"):
        script.generate_programs_api(
            "test-model",
            "system",
            "user",
            generation=script.GenerationSettings(
                n=1,
                temperature=0.8,
                top_p=0.95,
                max_new_tokens=128,
            ),
            api_base="https://example.invalid/v1",
            api_key=None,
        )


def test_generate_programs_api_missing_openai_points_to_api_extra(monkeypatch):
    import sys

    script = load_script_module("generate_pilot_programs")
    monkeypatch.setitem(sys.modules, "openai", None)

    with pytest.raises(ImportError, match=r"synthstats\[api\]"):
        script.generate_programs_api(
            "test-model",
            "system",
            "user",
            generation=script.GenerationSettings(
                n=1,
                temperature=0.8,
                top_p=0.95,
                max_new_tokens=128,
            ),
            api_base="https://example.invalid/v1",
            api_key="fake-key-1",
        )
