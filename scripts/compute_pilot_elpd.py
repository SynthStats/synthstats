#!/usr/bin/env python3
"""Compute in-sample predictive rewards for pilot LazyPPL ATIF trajectories.

For each pilot-*.json ATIF file:
  1. Extract Haskell code from the tool_call
  2. Load matching pilot task JSON (columns + observations)
  3. Run MH inference via LazyPPLExecutor.execute_for_loglik()
  4. Compute mean in-sample log predictive density across observations
  5. Compute reward = exp(0.1 * density) clamped to [1e-8, 1.0]
  6. Update the ATIF file's final_metrics.extra
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from _utils import MIN_REWARD, extract_code, pilot_data_dir, write_json_atomic
from sft_to_atif import validate_atif_record

from synthstats.core.reward_semantics import annotate_reward_info

logger = logging.getLogger(__name__)

# env name mapping: filename fragment -> pilot task JSON stem
ENV_MAP = {
    "dugongs": "lazyppl_pilot_dugongs",
    "peregrines": "lazyppl_pilot_peregrines",
    "eight_schools": "lazyppl_pilot_eight_schools",
    "surgical": "lazyppl_pilot_surgical",
}

BETA = 0.1
BURNIN = 1000
THINNING = 10
NDRAWS = 200
TIMEOUT_S = 120


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_atif_dir() -> Path:
    return project_root() / "data" / "atif_trajectories"


def default_pilot_data_dir() -> Path:
    return pilot_data_dir()


def load_lazyppl_executor() -> Any:
    sys.path.insert(0, str(project_root() / "src"))
    from synthstats.executors.lazyppl_sandbox import LazyPPLExecutor

    return LazyPPLExecutor()


def env_from_filename(name: str) -> str | None:
    # pilot-dugongs-0001.json -> dugongs
    # pilot-eight_schools-0003.json -> eight_schools
    parts = name.replace(".json", "").split("-")
    # parts[0] = "pilot", parts[-1] = "0001", middle = env name
    if len(parts) < 3:
        return None
    return "-".join(parts[1:-1])


def compute_in_sample_log_predictive_density(log_lik: np.ndarray) -> float:
    """Return mean in-sample log posterior predictive density across observations.

    log_lik shape: (n_draws, n_obs)
    Each column uses log-mean-exp over draws fitted to these same observations.
    """
    # log-mean-exp per observation column: log(1/S * sum_s exp(ll_s,n))
    n_draws = log_lik.shape[0]
    # use logsumexp for numerical stability
    from scipy.special import logsumexp

    lme = logsumexp(log_lik, axis=0) - np.log(n_draws)  # shape (n_obs,)
    return float(np.mean(lme))


def compute_reward(density: float) -> float:
    raw = np.exp(BETA * density)
    return float(np.clip(raw, MIN_REWARD, 1.0))


def _update_final_metrics(
    atif: dict[str, Any],
    *,
    reward: float,
    reward_source: str,
    extra: dict[str, Any] | None = None,
) -> None:
    final_metrics = atif.setdefault("final_metrics", {}).setdefault("extra", {})
    final_metrics.pop("elpd", None)
    final_metrics.pop("elpd_status", None)
    annotated = annotate_reward_info(
        {
            "reward": reward,
            "reward_source": reward_source,
        },
        reward_source=reward_source,
    )
    final_metrics.update(annotated)
    if extra:
        final_metrics.update(extra)
    validate_atif_record(atif)


def mark_failure(atif: dict[str, Any], *, status: str) -> None:
    reward_source = (
        "verifier_missing_program"
        if status == "missing_code"
        else "validation.in_sample_log_predictive_density"
    )
    _update_final_metrics(
        atif,
        reward=MIN_REWARD,
        reward_source=reward_source,
        extra={
            "in_sample_log_predictive_density": None,
            "in_sample_log_predictive_density_status": status,
        },
    )


def main(
    *,
    atif_dir: Path | None = None,
    pilot_data_dir: Path | None = None,
    executor: Any | None = None,
) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    resolved_atif_dir = atif_dir or default_atif_dir()
    resolved_pilot_data_dir = pilot_data_dir

    pilot_files = sorted(resolved_atif_dir.glob("pilot-*.json"))
    if not pilot_files:
        logger.error("No pilot-*.json files found in %s", resolved_atif_dir)
        return 1

    resolved_executor = executor or load_lazyppl_executor()

    logger.info("Found %d pilot ATIF files", len(pilot_files))

    # preload task data
    task_cache: dict[str, dict] = {}

    results: list[tuple[str, str, float | None, float, float]] = []
    succeeded = 0
    failed = 0

    for i, fpath in enumerate(pilot_files):
        atif = json.loads(fpath.read_text())
        session_id = atif.get("session_id", fpath.stem)

        code = extract_code(atif)
        if not code:
            logger.warning("[%d] %s: no code found, skipping", i, session_id)
            mark_failure(atif, status="missing_code")
            write_json_atomic(fpath, atif)
            failed += 1
            continue

        env_name = env_from_filename(fpath.name)
        if env_name not in ENV_MAP:
            logger.warning("[%d] %s: unknown env '%s'", i, session_id, env_name)
            mark_failure(atif, status="unknown_env")
            write_json_atomic(fpath, atif)
            failed += 1
            continue

        task_key = ENV_MAP[env_name]
        if task_key not in task_cache:
            if resolved_pilot_data_dir is None:
                resolved_pilot_data_dir = default_pilot_data_dir()
            task_path = resolved_pilot_data_dir / f"{task_key}.json"
            if not task_path.exists():
                logger.error("Task data not found: %s", task_path)
                mark_failure(atif, status="missing_task_data")
                write_json_atomic(fpath, atif)
                failed += 1
                continue
            task_cache[task_key] = json.loads(task_path.read_text(encoding="utf-8"))

        task = task_cache[task_key]
        columns = task["columns"]
        data_rows = task["observations"]

        logger.info(
            "[%d/%d] %s  env=%s  n_obs=%d ...",
            i + 1,
            len(pilot_files),
            session_id,
            env_name,
            len(data_rows),
        )

        t0 = time.time()
        log_lik = resolved_executor.execute_for_loglik(
            code,
            columns,
            data_rows,
            burnin=BURNIN,
            thinning=THINNING,
            ndraws=NDRAWS,
            timeout_s=TIMEOUT_S,
        )
        elapsed = time.time() - t0

        if log_lik is None:
            logger.warning("  FAILED (%.1fs) - MH inference returned None", elapsed)
            mark_failure(atif, status="failed")
            write_json_atomic(fpath, atif)
            failed += 1
            results.append((session_id, env_name, None, MIN_REWARD, elapsed))
            continue

        # sanity: check for NaN/Inf
        finite_frac = np.isfinite(log_lik).mean()
        if finite_frac < 0.5:
            logger.warning(
                "  %.0f%% of log_lik entries are non-finite, treating as failure",
                (1 - finite_frac) * 100,
            )
            mark_failure(atif, status="non_finite")
            write_json_atomic(fpath, atif)
            failed += 1
            results.append((session_id, env_name, None, MIN_REWARD, elapsed))
            continue

        # replace non-finite with large negative
        log_lik = np.where(np.isfinite(log_lik), log_lik, -1e6)

        density = compute_in_sample_log_predictive_density(log_lik)
        reward = compute_reward(density)

        logger.info(
            "  OK (%.1fs) shape=%s in_sample_log_predictive_density=%.4f reward=%.6f",
            elapsed,
            log_lik.shape,
            density,
            reward,
        )

        _update_final_metrics(
            atif,
            reward=reward,
            reward_source="validation.in_sample_log_predictive_density",
            extra={
                "in_sample_log_predictive_density": density,
                "in_sample_log_predictive_density_status": "ok",
                "log_lik_shape": list(log_lik.shape),
            },
        )
        write_json_atomic(fpath, atif)

        succeeded += 1
        results.append((session_id, env_name, density, reward, elapsed))

    # summary
    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "SUMMARY: %d succeeded, %d failed out of %d total", succeeded, failed, len(pilot_files)
    )
    logger.info("=" * 70)

    if results:
        logger.info("")
        logger.info(
            "%-40s %-15s %10s %10s %8s",
            "session_id",
            "env",
            "in_sample_log_predictive_density",
            "reward",
            "time_s",
        )
        logger.info("-" * 90)
        density_vals: list[float] = []
        reward_vals: list[float] = []
        for sid, env, density_value, reward, elapsed in results:
            density_str = f"{density_value:.4f}" if density_value is not None else "FAIL"
            logger.info("%-40s %-15s %10s %10.6f %8.1f", sid, env, density_str, reward, elapsed)
            if density_value is not None:
                density_vals.append(density_value)
                reward_vals.append(reward)

        if density_vals:
            logger.info("")
            logger.info(
                "In-sample log predictive density: mean=%.4f std=%.4f min=%.4f max=%.4f",
                np.mean(density_vals),
                np.std(density_vals),
                np.min(density_vals),
                np.max(density_vals),
            )
            logger.info(
                "Reward: mean=%.6f  std=%.6f  min=%.6f  max=%.6f",
                np.mean(reward_vals),
                np.std(reward_vals),
                np.min(reward_vals),
                np.max(reward_vals),
            )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
