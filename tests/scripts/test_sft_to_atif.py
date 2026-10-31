from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import _utils as utils
from conftest import load_script_module


def test_cli_requires_input_path_before_creating_output(tmp_path):
    output_dir = tmp_path / "atif"
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "sft_to_atif.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--output-dir", str(output_dir)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 2
    assert "--input-path" in result.stderr
    assert "required" in result.stderr
    assert not output_dir.exists()


class ByteTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))


def test_pymc_record_uses_its_selected_runtime():
    script = load_script_module("sft_to_atif")
    completion = (
        '<submit_program language="pymc">'
        "import pymc as pm\n"
        "def model(data):\n"
        "    with pm.Model() as result:\n"
        '        pm.Normal("obs", 0, 1, observed=data["y"])\n'
        "    return result\n"
        "</submit_program>"
    )

    atif = script.build_atif_record(
        1, "Model the observations in y.", completion, tokenizer=ByteTokenizer()
    )

    system_message = atif["steps"][0]["message"].lower()
    assert "pymc" in system_message
    assert "lazyppl" not in system_message
    assert "haskell" not in system_message
    assert atif["final_metrics"]["extra"]["reward_source"] == "heuristic.runtime_import"
    assert atif["final_metrics"]["extra"]["reward"] == 1.0


def test_build_record_preserves_reasoning_content():
    script = load_script_module("sft_to_atif")
    prompt = "summarize data"
    completion = (
        "<think>latent trace</think>"
        '<submit_program language="haskell">module Submission where</submit_program>'
    )

    atif = script.build_atif_record(7, prompt, completion, tokenizer=ByteTokenizer())

    agent_step = atif["steps"][2]
    assert atif["session_id"] == "sft-lazyppl-0007"
    assert atif["schema_version"] == utils.ATIF_SCHEMA_VERSION
    assert atif["agent"] == utils.build_atif_agent(script.MODEL_NAME)
    assert agent_step["reasoning_content"] == "latent trace"
    assert "logprobs" not in agent_step["metrics"]
    assert agent_step["tool_calls"][0]["arguments"]["language"] == "haskell"


def test_tokenize_text_uses_tokenizer_when_present():
    script = load_script_module("sft_to_atif")

    class FakeTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            assert text == "hello world"
            assert add_special_tokens is False
            return [10, 20]

    assert script.tokenize_text(FakeTokenizer(), "hello world") == [10, 20]


def test_build_record_supports_dataset_rows_and_preserves_shared_metadata():
    script = load_script_module("sft_to_atif")
    record = {
        "id": "completion-001",
        "prompt_id": "prompt-001",
        "env_id": "dugongs",
        "split": "train",
        "prompt": "Write a model for grouped counts.",
        "completion": "```python\nwith pm.Model():\n    x = 1\n```",
        "pymc_version": "5.20.0",
        "trace_id": "completion-001:validation",
        "candidate_outputs_json": json.dumps(["```python\nx = 2\n```", "```python\nx = 3\n```"]),
        "provenance_json": json.dumps(
            {
                "id": "completion-001",
                "prompt_id": "prompt-001",
                "provenance": {"runtime_name": "pymc", "runtime_version": "5.20.0"},
                "metadata": {"source": "dataset"},
            }
        ),
        "env_name": "Dugongs",
        "data_ref": "data/dugongs.parquet",
        "trace_json": json.dumps(
            {
                "schema_version": "ppl.platform.record.v0",
                "trace_id": "completion-001:validation",
                "task_id": "dugongs",
                "trial_id": "completion-001",
                "steps": [],
                "artifacts": {},
                "metadata": {},
            }
        ),
        "validation_json": json.dumps(
            {
                "ok": True,
                "message": "passed",
                "semantic_metric": "avg_holdout_logp",
                "semantic_score": -0.123,
                "semantic_status": "pass",
                "evidence_kind": "verifier",
                "truth_owner": "verifier",
            }
        ),
    }

    atif = script.build_atif_record(
        7,
        record["prompt"],
        record["completion"],
        tokenizer=ByteTokenizer(),
        record=record,
    )

    assert atif["session_id"] == "sft-pymc-0007"
    assert atif["steps"][2]["tool_calls"][0]["arguments"]["language"] == "pymc"
    assert atif["final_metrics"]["extra"]["reward"] == -0.123
    assert atif["final_metrics"]["extra"]["reward_source"] == "validation.semantic_score"
    assert atif["final_metrics"]["extra"]["evidence_kind"] == "verifier"
    assert atif["final_metrics"]["extra"]["truth_owner"] == "verifier"
    assert atif["extra"]["source"] == "synthstats.dataset_row"
    assert atif["extra"]["dataset_row"]["id"] == "completion-001"
    assert atif["extra"]["shared_generation_record"]["record_id"] == "completion-001"
    assert atif["extra"]["shared_generation_record"]["candidate_outputs"] == [
        "```python\nx = 2\n```",
        "```python\nx = 3\n```",
    ]
    assert atif["extra"]["shared_trial_record"]["trace_id"] == "completion-001:validation"
    assert atif["extra"]["shared_trace_record"]["trace_id"] == "completion-001:validation"


def test_build_record_recovers_shared_trace_from_validation_trace_fallback():
    script = load_script_module("sft_to_atif")
    record = {
        "id": "completion-fallback-001",
        "prompt_id": "prompt-fallback-001",
        "env_id": "dugongs",
        "split": "train",
        "prompt": "Write a model for grouped counts.",
        "completion": "```python\nwith pm.Model():\n    x = 1\n```",
        "pymc_version": "5.20.0",
        "trace_id": "completion-fallback-001:validation",
        "provenance_json": json.dumps(
            {
                "id": "completion-fallback-001",
                "prompt_id": "prompt-fallback-001",
                "provenance": {"runtime_name": "pymc", "runtime_version": "5.20.0"},
                "metadata": {"source": "dataset"},
            }
        ),
        "validation_json": json.dumps(
            {
                "ok": True,
                "message": "passed",
                "semantic_metric": "psis_loo",
                "semantic_score": 0.81,
                "semantic_status": "pass",
                "evidence_kind": "verifier",
                "truth_owner": "verifier",
                "trace": {
                    "steps": [
                        {
                            "attempt": 0,
                            "phase": "runtime",
                            "ok": True,
                            "message": "ok",
                            "evidence_kind": "verifier",
                            "truth_owner": "verifier",
                            "semantic_status": "not_scored",
                        },
                        {
                            "attempt": 0,
                            "phase": "semantic_validation",
                            "ok": True,
                            "message": "pass",
                            "evidence_kind": "verifier",
                            "truth_owner": "verifier",
                            "semantic_status": "pass",
                            "metadata": {
                                "metric_name": "psis_loo",
                                "metric_value": 0.81,
                            },
                        },
                    ]
                },
            }
        ),
    }

    atif = script.build_atif_record(
        8,
        record["prompt"],
        record["completion"],
        tokenizer=ByteTokenizer(),
        record=record,
    )

    assert atif["extra"]["shared_trace_record"]["trace_id"] == (
        "completion-fallback-001:validation"
    )
    assert atif["extra"]["shared_trace_record"]["metadata"]["trace_origin"] == (
        "validation_json.trace"
    )
    assert atif["extra"]["shared_trace_record"]["steps"][0]["evidence_kind"] == ("verifier")
    assert atif["extra"]["shared_trace_record"]["steps"][0]["metadata"]["phase"] == ("runtime")
    assert atif["extra"]["shared_trace_record"]["steps"][1]["semantic_status"] == ("pass")


def test_build_record_enriches_empty_trace_json_from_validation_trace() -> None:
    script = load_script_module("sft_to_atif")
    record = {
        "id": "completion-empty-trace-001",
        "prompt_id": "prompt-empty-trace-001",
        "env_id": "dugongs",
        "split": "train",
        "prompt": "Write a model for grouped counts.",
        "completion": "```python\nwith pm.Model():\n    x = 1\n```",
        "pymc_version": "5.20.0",
        "trace_id": "completion-empty-trace-001:validation",
        "provenance_json": json.dumps(
            {
                "id": "completion-empty-trace-001",
                "prompt_id": "prompt-empty-trace-001",
                "provenance": {"runtime_name": "pymc", "runtime_version": "5.20.0"},
                "metadata": {"source": "dataset"},
            }
        ),
        "trace_json": json.dumps(
            {
                "schema_version": "ppl.platform.record.v0",
                "trace_id": "completion-empty-trace-001:validation",
                "task_id": "dugongs",
                "trial_id": "completion-empty-trace-001",
                "steps": [],
                "artifacts": {"outcome": "validated"},
                "metadata": {"source": "stale_trace_json"},
            }
        ),
        "validation_json": json.dumps(
            {
                "ok": True,
                "message": "passed",
                "semantic_metric": "psis_loo",
                "semantic_score": 0.81,
                "semantic_status": "pass",
                "evidence_kind": "verifier",
                "truth_owner": "verifier",
                "trace": {
                    "steps": [
                        {
                            "attempt": 0,
                            "phase": "runtime",
                            "ok": True,
                            "message": "ok",
                            "evidence_kind": "verifier",
                            "truth_owner": "verifier",
                            "semantic_status": "not_scored",
                        }
                    ]
                },
            }
        ),
    }

    atif = script.build_atif_record(
        9,
        record["prompt"],
        record["completion"],
        tokenizer=ByteTokenizer(),
        record=record,
    )

    assert atif["extra"]["shared_trace_record"]["metadata"]["source"] == ("stale_trace_json")
    assert atif["extra"]["shared_trace_record"]["metadata"]["trace_origin"] == (
        "trace_json.empty+validation_json.trace"
    )
    assert atif["extra"]["shared_trace_record"]["artifacts"]["outcome"] == ("validated")
    assert atif["extra"]["shared_trace_record"]["steps"][0]["metadata"]["phase"] == ("runtime")
    assert atif["extra"]["shared_trace_record"]["steps"][0]["evidence_kind"] == ("verifier")


def test_build_record_promotes_trace_json_taxonomy_from_metadata() -> None:
    script = load_script_module("sft_to_atif")
    record = {
        "id": "completion-trace-001",
        "prompt_id": "prompt-trace-001",
        "env_id": "dugongs",
        "split": "train",
        "prompt": "Write a model for grouped counts.",
        "completion": "```python\nwith pm.Model():\n    x = 1\n```",
        "pymc_version": "5.20.0",
        "trace_id": "completion-trace-001:validation",
        "provenance_json": json.dumps(
            {
                "id": "completion-trace-001",
                "prompt_id": "prompt-trace-001",
                "provenance": {"runtime_name": "pymc", "runtime_version": "5.20.0"},
                "metadata": {"source": "dataset"},
            }
        ),
        "trace_json": json.dumps(
            {
                "schema_version": "ppl.platform.record.v0",
                "trace_id": "completion-trace-001:validation",
                "task_id": "dugongs",
                "trial_id": "completion-trace-001",
                "steps": [
                    {
                        "step_id": "step-0000",
                        "role": "validator",
                        "content": "ok",
                        "metadata": {
                            "phase": "runtime",
                            "ok": True,
                            "evidence_kind": "verifier",
                            "truth_owner": "verifier",
                            "semantic_status": "not_scored",
                        },
                    }
                ],
                "artifacts": {"outcome": "validated"},
                "metadata": {"source": "trace_json_metadata"},
            }
        ),
        "validation_json": json.dumps(
            {
                "ok": True,
                "message": "passed",
                "semantic_metric": "psis_loo",
                "semantic_score": 0.81,
                "semantic_status": "pass",
                "evidence_kind": "verifier",
                "truth_owner": "verifier",
            }
        ),
    }

    atif = script.build_atif_record(
        10,
        record["prompt"],
        record["completion"],
        tokenizer=ByteTokenizer(),
        record=record,
    )

    assert atif["extra"]["shared_trace_record"]["metadata"]["source"] == ("trace_json_metadata")
    assert atif["extra"]["shared_trace_record"]["steps"][0]["metadata"]["phase"] == ("runtime")
    assert atif["extra"]["shared_trace_record"]["steps"][0]["evidence_kind"] is None
    assert atif["extra"]["shared_trace_record"]["steps"][0]["truth_owner"] is None
    assert atif["extra"]["shared_trace_record"]["steps"][0]["semantic_status"] is None
    assert atif["extra"]["shared_trace_record"]["steps"][0]["metadata"]["evidence_kind"] == (
        "verifier"
    )
    assert atif["extra"]["shared_trace_record"]["steps"][0]["metadata"]["truth_owner"] == (
        "verifier"
    )
    assert atif["extra"]["shared_trace_record"]["steps"][0]["metadata"]["semantic_status"] == (
        "not_scored"
    )


def test_build_record_infers_verifier_semantics_from_validation_ok():
    script = load_script_module("sft_to_atif")
    record = {
        "prompt": "Write a model.",
        "completion": "```python\nwith pm.Model():\n    pass\n```",
        "pymc_version": "5.20.0",
        "validation_json": json.dumps({"ok": True, "message": "passed"}),
    }

    atif = script.build_atif_record(
        3,
        record["prompt"],
        record["completion"],
        tokenizer=ByteTokenizer(),
        record=record,
    )

    assert atif["final_metrics"]["extra"]["reward"] == 1.0
    assert atif["final_metrics"]["extra"]["reward_source"] == "validation.ok"
    assert atif["final_metrics"]["extra"]["evidence_kind"] == "verifier"
    assert atif["final_metrics"]["extra"]["truth_owner"] == "verifier"


def test_build_record_infers_heuristic_semantics_from_runtime_import_fallback():
    script = load_script_module("sft_to_atif")

    atif = script.build_atif_record(
        4,
        "Write a model.",
        (
            '<submit_program language="haskell">'
            "module Submission where\nimport LazyPPL\n"
            "</submit_program>"
        ),
        tokenizer=ByteTokenizer(),
    )

    assert atif["final_metrics"]["extra"]["reward"] == 1.0
    assert atif["final_metrics"]["extra"]["reward_source"] == "heuristic.runtime_import"
    assert atif["final_metrics"]["extra"]["evidence_kind"] == "heuristic"
    assert atif["final_metrics"]["extra"]["truth_owner"] == "heuristic"


def test_build_record_validates_against_harbor_atif_schema():
    from harbor.models.trajectories.trajectory import Trajectory

    script = load_script_module("sft_to_atif")
    atif = script.build_atif_record(
        4,
        "Write a model.",
        (
            '<submit_program language="haskell">'
            "module Submission where\nimport LazyPPL\n"
            "</submit_program>"
        ),
        tokenizer=ByteTokenizer(),
    )

    validated = Trajectory.model_validate(atif)
    assert validated.session_id == "sft-lazyppl-0004"
    assert atif["extra"]["training_data_kind"] == "supervised"
    assert atif["extra"]["behavior_logprobs_available"] is False
    assert atif["final_metrics"]["extra"]["reward"] == 1.0
    assert atif["final_metrics"]["extra"]["reward_source"] == "heuristic.runtime_import"


def test_load_sft_records_skips_blank_lines_and_preserves_record_indices(tmp_path):
    script = load_script_module("sft_to_atif")
    input_path = tmp_path / "train.jsonl"
    input_path.write_text(
        "\n".join(
            [
                "",
                json.dumps(
                    {
                        "prompt": "first",
                        "completion": (
                            '<submit_program language="haskell">module First where</submit_program>'
                        ),
                    }
                ),
                "",
                json.dumps(
                    {
                        "prompt": "second",
                        "completion": (
                            '<submit_program language="haskell">'
                            "module Second where"
                            "</submit_program>"
                        ),
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    records, errors = script.load_sft_records(input_path)

    assert errors == 0
    assert [record_index for record_index, _ in records] == [0, 1]
    assert [record["prompt"] for _, record in records] == ["first", "second"]


def test_main_writes_manifest_and_records(tmp_path, monkeypatch):
    script = load_script_module("sft_to_atif")
    input_path = tmp_path / "train.jsonl"
    output_dir = tmp_path / "atif_trajectories"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt": "summarize dugongs",
                        "completion": (
                            "<think>latent trace</think>"
                            '<submit_program language="haskell">'
                            "module Submission where"
                            "</submit_program>"
                        ),
                    }
                ),
                json.dumps(
                    {
                        "prompt": "summarize peregrines",
                        "completion": (
                            '<submit_program language="haskell">module Birds where</submit_program>'
                        ),
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(script, "load_tokenizer", lambda: ByteTokenizer())

    assert script.main(input_path=input_path, output_dir=output_dir) == 0

    manifest = json.loads((output_dir / "manifest.json").read_text())
    first_record = json.loads((output_dir / "sft-lazyppl-0000.json").read_text())

    assert manifest["schema_version"] == utils.ATIF_SCHEMA_VERSION
    assert manifest["count"] == 2
    assert manifest["files"][0]["session_id"] == "sft-lazyppl-0000"
    assert manifest["files"][0]["reward_source"] == "heuristic.runtime_import"
    assert manifest["files"][0]["evidence_kind"] == "heuristic"
    assert manifest["files"][0]["truth_owner"] == "heuristic"
    assert first_record["steps"][2]["reasoning_content"] == "latent trace"
    assert first_record["steps"][2]["tool_calls"][0]["function_name"] == "submit_program"


def test_main_returns_partial_failure_code(tmp_path, monkeypatch):
    script = load_script_module("sft_to_atif")
    input_path = tmp_path / "train.jsonl"
    output_dir = tmp_path / "atif_trajectories"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt": "valid",
                        "completion": (
                            '<submit_program language="haskell">module Valid where</submit_program>'
                        ),
                    }
                ),
                json.dumps(
                    {
                        "prompt": "invalid",
                        "completion": "<think>missing submit tag</think>",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(script, "load_tokenizer", lambda: ByteTokenizer())

    assert script.main(input_path=input_path, output_dir=output_dir) == 1

    manifest = json.loads((output_dir / "manifest.json").read_text())

    assert manifest["count"] == 1
    assert manifest["files"] == [
        {
            "session_id": "sft-lazyppl-0000",
            "file": "sft-lazyppl-0000.json",
            "reward": 0.0,
            "reward_source": "heuristic.runtime_import",
            "evidence_kind": "heuristic",
            "truth_owner": "heuristic",
        }
    ]
    assert not (output_dir / "sft-lazyppl-0001.json").exists()


def test_main_skips_malformed_jsonl_lines(tmp_path, monkeypatch):
    script = load_script_module("sft_to_atif")
    input_path = tmp_path / "train.jsonl"
    output_dir = tmp_path / "atif_trajectories"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt": "valid first",
                        "completion": (
                            '<submit_program language="haskell">'
                            "module First where\nimport LazyPPL"
                            "</submit_program>"
                        ),
                    }
                ),
                '{"prompt": "broken", "completion": ',
                json.dumps(
                    {
                        "prompt": "valid third",
                        "completion": (
                            '<submit_program language="haskell">'
                            "module Third where\nimport LazyPPL"
                            "</submit_program>"
                        ),
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(script, "load_tokenizer", lambda: ByteTokenizer())

    assert script.main(input_path=input_path, output_dir=output_dir) == 1

    manifest = json.loads((output_dir / "manifest.json").read_text())

    assert manifest["count"] == 2
    assert manifest["files"] == [
        {
            "session_id": "sft-lazyppl-0000",
            "file": "sft-lazyppl-0000.json",
            "reward": 1.0,
            "reward_source": "heuristic.runtime_import",
            "evidence_kind": "heuristic",
            "truth_owner": "heuristic",
        },
        {
            "session_id": "sft-lazyppl-0002",
            "file": "sft-lazyppl-0002.json",
            "reward": 1.0,
            "reward_source": "heuristic.runtime_import",
            "evidence_kind": "heuristic",
            "truth_owner": "heuristic",
        },
    ]
    assert not (output_dir / "sft-lazyppl-0001.json").exists()
