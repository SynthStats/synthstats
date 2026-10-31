from __future__ import annotations

import json
import subprocess

import _utils as utils
import pytest


def test_write_json_atomic_replaces_existing_contents(tmp_path):
    target = tmp_path / "payload.json"
    target.write_text('{"old": true}')

    utils.write_json_atomic(target, {"new": 2})

    assert json.loads(target.read_text()) == {"new": 2}
    assert list(tmp_path.glob("*.tmp")) == []


def test_build_atif_agent_returns_shared_metadata():
    agent = utils.build_atif_agent("test-model")

    assert agent == {
        "name": utils.ATIF_AGENT_NAME,
        "version": utils.ATIF_AGENT_VERSION,
        "model_name": "test-model",
    }


def test_typecheck_uses_selected_backend_and_cleans_temp_source(tmp_path, monkeypatch):
    lazyppl_dir = tmp_path / "lazyppl"
    (lazyppl_dir / "src").mkdir(parents=True)
    monkeypatch.setenv("LAZYPPL_DEV_DIR", str(lazyppl_dir))

    observed: dict[str, object] = {}

    def fake_run(args, **kwargs):
        observed["args"] = args
        observed["cwd"] = kwargs["cwd"]
        observed["timeout"] = kwargs["timeout"]
        src_path = lazyppl_dir / "src" / "RewardCheck.hs"
        observed["source"] = src_path.read_text()
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="Ok, one module loaded\nmodules loaded",
            stderr="",
        )

    monkeypatch.setattr(utils.subprocess, "run", fake_run)

    ok, output = utils.typecheck(
        "module OldName where\nvalue = 1",
        "RewardCheck",
        backend="cabal",
        timeout_s=7,
    )

    assert ok is True
    assert "modules loaded" in output
    assert observed["cwd"] == lazyppl_dir
    assert ":load RewardCheck" in observed["args"][2]
    assert "cabal repl --repl-options='-fno-code'" in observed["args"][2]
    assert observed["timeout"] == 7
    assert observed["source"].startswith("module RewardCheck where")
    assert not (lazyppl_dir / "src" / "RewardCheck.hs").exists()


def test_typecheck_returns_timeout_on_subprocess_timeout(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    monkeypatch.setenv("LAZYPPL_DEV_DIR", str(tmp_path))

    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs["timeout"])

    monkeypatch.setattr(utils.subprocess, "run", fake_run)

    ok, output = utils.typecheck("value = 1", timeout_s=5)

    assert ok is False
    assert output == "timeout"


def test_lazyppl_and_pilot_data_dirs_require_env_vars(monkeypatch):
    monkeypatch.delenv("LAZYPPL_DEV_DIR", raising=False)
    monkeypatch.delenv("PILOT_DATA_DIR", raising=False)

    with pytest.raises(RuntimeError, match="LAZYPPL_DEV_DIR"):
        utils.lazyppl_dir()

    with pytest.raises(RuntimeError, match="PILOT_DATA_DIR"):
        utils.pilot_data_dir()
