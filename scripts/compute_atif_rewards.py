#!/usr/bin/env python3
"""Compute real GHC compilation rewards for ATIF trajectory files.

Reads each ATIF JSON file from data/atif_trajectories/, extracts the Haskell
program from tool_calls, type-checks via cabal repl, and updates
final_metrics.extra.reward to 1.0 (compiles) or 1e-8 (doesn't).
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from _utils import MIN_REWARD, extract_code, typecheck, write_json_atomic
from sft_to_atif import validate_atif_record

from synthstats.core.reward_semantics import annotate_reward_info

logger = logging.getLogger(__name__)

REWARD_COMPILES = 1.0
REWARD_FAILS = MIN_REWARD
TIMEOUT_S = 30


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_atif_dir() -> Path:
    return project_root() / "data" / "atif_trajectories"


def load_atif(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


_SESSION_RUNTIME_PATTERN = re.compile(r"^sft-([a-z0-9_-]+)-\d+$")


def _atif_runtime_name(atif: dict[str, Any], path: Path) -> str | None:
    for step in atif.get("steps", []):
        if step.get("source") != "agent":
            continue
        for tc in step.get("tool_calls", []):
            args = tc.get("arguments", {})
            language = args.get("language")
            if isinstance(language, str) and language.strip():
                return language.strip().lower()

    for candidate in (atif.get("session_id"), path.stem):
        if not isinstance(candidate, str):
            continue
        match = _SESSION_RUNTIME_PATTERN.match(candidate)
        if match is not None:
            return match.group(1).lower()
    return None


def _supports_haskell_reward(runtime_name: str | None) -> bool:
    return runtime_name in {"haskell", "lazyppl"}


def _set_compilation_reward(
    atif: dict[str, Any],
    *,
    reward: float,
    reward_source: str,
) -> None:
    final_metrics = atif.setdefault("final_metrics", {}).setdefault("extra", {})
    annotated = annotate_reward_info(
        {
            "reward": reward,
            "reward_source": reward_source,
        },
        reward_source=reward_source,
    )
    final_metrics.update(annotated)
    validate_atif_record(atif)


def main(*, atif_dir: Path | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    resolved_atif_dir = atif_dir or default_atif_dir()
    if not resolved_atif_dir.exists():
        logger.error("ATIF directory not found: %s", resolved_atif_dir)
        return 1

    files = sorted(resolved_atif_dir.glob("sft-*.json"))
    logger.info("Processing %d ATIF files...", len(files))

    compiles = 0
    fails = 0
    errors = 0
    skipped = 0

    for i, f in enumerate(files):
        atif = load_atif(f)
        runtime_name = _atif_runtime_name(atif, f)
        if not _supports_haskell_reward(runtime_name):
            skipped += 1
            logger.info("  [%d] %s: skipped unsupported runtime %r", i, f.name, runtime_name)
            continue
        code = extract_code(atif)

        if not code:
            logger.warning("  [%d] %s: no code found", i, f.name)
            _set_compilation_reward(
                atif,
                reward=REWARD_FAILS,
                reward_source="verifier_missing_program",
            )
            errors += 1
        else:
            # use unique module name to avoid collisions
            module_name = f"RewardCheck{i:04d}"
            ok, _ = typecheck(
                code,
                module_name,
                backend="cabal",
                timeout_s=TIMEOUT_S,
            )
            _set_compilation_reward(
                atif,
                reward=REWARD_COMPILES if ok else REWARD_FAILS,
                reward_source="validation.compilation",
            )
            if ok:
                compiles += 1
            else:
                fails += 1

            if (i + 1) % 10 == 0:
                logger.info("  [%d/%d] compiles=%d fails=%d", i + 1, len(files), compiles, fails)

        write_json_atomic(f, atif)

    total = compiles + fails + errors
    logger.info("")
    logger.info(
        "Done: %d/%d compile (%.0f%%), %d fail, %d no-code",
        compiles,
        total,
        100 * compiles / max(total, 1),
        fails,
        errors,
    )
    if skipped:
        logger.info("Skipped %d unsupported-runtime files", skipped)
    return 0 if fails == 0 and errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
