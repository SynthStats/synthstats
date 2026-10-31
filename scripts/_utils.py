"""Shared utilities for scripts/."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ATIF_SCHEMA_VERSION = "ATIF-v1.6"
ATIF_AGENT_NAME = "synthstats"
ATIF_AGENT_VERSION = "0.1.0"
MIN_REWARD = 1e-8

SAFE_ENV_VARS = {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "CABAL_DIR", "GHC_PACKAGE_PATH"}

TYPECHECK_TIMEOUT_S = 30
TYPECHECK_BACKENDS: dict[str, str] = {
    "stack": "stack repl --ghc-options='-fno-code' 2>&1",
    "cabal": "cabal repl --repl-options='-fno-code' 2>&1",
}


def lazyppl_dir() -> Path:
    raw = os.environ.get("LAZYPPL_DEV_DIR")
    if not raw:
        raise RuntimeError("LAZYPPL_DEV_DIR environment variable is required")
    return Path(raw)


def pilot_data_dir() -> Path:
    raw = os.environ.get("PILOT_DATA_DIR")
    if not raw:
        raise RuntimeError("PILOT_DATA_DIR environment variable is required")
    return Path(raw)


def build_atif_agent(model_name: str) -> dict[str, str]:
    """Return shared ATIF agent metadata with the given model name."""
    return {
        "name": ATIF_AGENT_NAME,
        "version": ATIF_AGENT_VERSION,
        "model_name": model_name,
    }


def write_json_atomic(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Atomically write JSON to disk using temp-file + rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f"{path.stem}-",
        suffix=".tmp",
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def extract_code(atif: dict) -> str | None:
    """Extract Haskell code from ATIF tool_calls."""
    for step in atif.get("steps", []):
        if step.get("source") != "agent":
            continue
        for tc in step.get("tool_calls", []):
            args = tc.get("arguments", {})
            code = args.get("code") or args.get("program")
            if code:
                return code
    return None


def typecheck(
    code: str,
    module_name: str = "Check",
    *,
    backend: Literal["stack", "cabal"] = "stack",
    timeout_s: int = TYPECHECK_TIMEOUT_S,
) -> tuple[bool, str]:
    """Type-check Haskell code via the selected repl backend."""
    if backend not in TYPECHECK_BACKENDS:
        raise ValueError(f"Unsupported typecheck backend: {backend}")
    if "module " not in code:
        code = f"module {module_name} where\n\n{code}"
    else:
        code = re.sub(r"module\s+\S+", f"module {module_name}", code, count=1)

    resolved_lazyppl_dir = lazyppl_dir()
    src_path = resolved_lazyppl_dir / "src" / f"{module_name}.hs"
    try:
        src_path.write_text(code, encoding="utf-8")

        env = {k: os.environ[k] for k in SAFE_ENV_VARS if k in os.environ}
        command = f"echo ':load {module_name}' | {TYPECHECK_BACKENDS[backend]}"
        try:
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    command,
                ],
                capture_output=True,
                text=True,
                cwd=resolved_lazyppl_dir,
                start_new_session=True,
                env=env,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"

        output = result.stdout + result.stderr
        ok = "Ok," in output and "modules loaded" in output
        return ok, output

    finally:
        if src_path.exists():
            src_path.unlink()
