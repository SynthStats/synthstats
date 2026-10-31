#!/usr/bin/env python
"""Persistent request supervisor with a fresh isolated interpreter per request.

JSON requests arrive on stdin and JSON responses leave on stdout.
Sentinel lines delimit messages; submitted code never runs in this process.
"""

from __future__ import annotations

import json
import subprocess
import sys

from synthstats.core.constants import PYMC_EXECUTOR_TIMEOUT_S
from synthstats.executors.isolated_python import run_isolated_python

READY_SENTINEL = "__SYNTHSTATS_WORKER_READY__"
END_REQUEST = "__SYNTHSTATS_END_REQUEST__"
END_RESPONSE = "__SYNTHSTATS_END_RESPONSE__"


def _execute_code(
    code: str, seed: int | None = None, timeout: float = PYMC_EXECUTOR_TIMEOUT_S
) -> dict:
    """Execute each request in a fresh, isolated interpreter."""
    if seed is not None:
        code = (
            f"import numpy as _np; _np.random.seed({int(seed)})\n"
            f"import random as _random; _random.seed({int(seed)})\n" + code
        )
    try:
        result = run_isolated_python(code, timeout=timeout)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": result.stderr if result.returncode else None,
            "returncode": result.returncode,
        }
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        return {
            "stdout": "",
            "stderr": str(exc),
            "error": str(exc),
            "returncode": 1,
        }


def main() -> None:
    # signal readiness to the parent
    print(READY_SENTINEL, flush=True)

    while True:
        lines: list[str] = []
        for raw in sys.stdin:
            line = raw.rstrip("\n")
            if line == END_REQUEST:
                break
            lines.append(line)
        else:
            # stdin closed (EOF) - exit cleanly
            break

        if not lines:
            break

        try:
            request = json.loads("\n".join(lines))
        except json.JSONDecodeError as exc:
            response = {
                "error": f"Invalid JSON: {exc}",
                "returncode": 1,
                "stdout": "",
                "stderr": "",
            }
        else:
            code = request.get("code", "")
            seed = request.get("seed")
            response = _execute_code(
                code, seed, float(request.get("timeout", PYMC_EXECUTOR_TIMEOUT_S))
            )

        print(json.dumps(response), flush=True)
        print(END_RESPONSE, flush=True)


if __name__ == "__main__":
    main()
