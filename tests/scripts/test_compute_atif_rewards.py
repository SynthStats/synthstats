from __future__ import annotations

import json

import compute_atif_rewards as script
from conftest import build_sft_record


def test_main_returns_error_code_when_data_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "project_root", lambda: tmp_path)

    assert script.main() == 1


def test_main_uses_cabal_typecheck_backend(tmp_path, monkeypatch):
    atif_dir = tmp_path / "data" / "atif_trajectories"
    atif_dir.mkdir(parents=True)
    atif_path = atif_dir / "sft-lazyppl-0000.json"
    atif_path.write_text(json.dumps(build_sft_record("module Submission where\nvalue = 1")))

    observed: dict[str, object] = {}

    def fake_typecheck(code, module_name, *, backend="stack", timeout_s=30):
        observed["code"] = code
        observed["module_name"] = module_name
        observed["backend"] = backend
        observed["timeout_s"] = timeout_s
        return True, "Ok, one module loaded\nmodules loaded"

    monkeypatch.setattr(script, "typecheck", fake_typecheck)

    assert script.main(atif_dir=atif_dir) == 0
    assert observed["module_name"] == "RewardCheck0000"
    assert observed["backend"] == "cabal"
    assert observed["timeout_s"] == script.TIMEOUT_S
    final_metrics = json.loads(atif_path.read_text())["final_metrics"]["extra"]
    assert final_metrics["reward"] == 1.0
    assert final_metrics["reward_source"] == "validation.compilation"
    assert final_metrics["evidence_kind"] == "verifier"
    assert final_metrics["truth_owner"] == "verifier"


def test_main_marks_no_code_records_as_failure(tmp_path):
    atif_dir = tmp_path / "data" / "atif_trajectories"
    atif_dir.mkdir(parents=True)
    atif_path = atif_dir / "sft-lazyppl-0001.json"
    atif_path.write_text(json.dumps(build_sft_record("")), encoding="utf-8")

    assert script.main(atif_dir=atif_dir) == 1

    final_metrics = json.loads(atif_path.read_text())["final_metrics"]["extra"]
    assert final_metrics["reward"] == script.REWARD_FAILS
    assert final_metrics["reward_source"] == "verifier_missing_program"
    assert final_metrics["evidence_kind"] == "verifier"
    assert final_metrics["truth_owner"] == "verifier"


def test_main_skips_unsupported_runtime_files(tmp_path, monkeypatch):
    atif_dir = tmp_path / "data" / "atif_trajectories"
    atif_dir.mkdir(parents=True)
    atif_path = atif_dir / "sft-pymc-0002.json"
    atif_path.write_text(
        json.dumps(
            {
                "session_id": "sft-pymc-0002",
                "steps": [
                    {
                        "source": "agent",
                        "tool_calls": [
                            {
                                "arguments": {
                                    "language": "pymc",
                                    "code": "import pymc as pm",
                                }
                            }
                        ],
                    }
                ],
                "final_metrics": {"reward": 0.5},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        script,
        "typecheck",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("typecheck should not run")),
    )

    assert script.main(atif_dir=atif_dir) == 0
    assert json.loads(atif_path.read_text())["final_metrics"]["reward"] == 0.5
