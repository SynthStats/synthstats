"""LikelihoodJudge - ELPD-based reward computation."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from synthstats.core.constants import LOG_REWARD_CLAMP_RANGE
from synthstats.core.reward_semantics import annotate_reward_info
from synthstats.core.runtime_registry import score_artifact_key_from_payload
from synthstats.core.types import Reward, Trajectory

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import arviz as az
    import numpy as np

_VALID_LOO_GATE_MODES = {"nonfinite", "strict"}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _summarize_loo_result(loo_result: Any, *, n_draws: int | None = None) -> dict[str, Any]:
    """Summarize the diagnostics that matter operationally for PSIS-LOO."""
    import numpy as np

    pareto_raw = getattr(loo_result, "pareto_k", None)
    pareto_k: Any
    if pareto_raw is not None:
        pareto_k = np.asarray(pareto_raw, dtype=np.float64).reshape(-1)
    else:
        pareto_k = np.array([], dtype=np.float64)
    finite_pareto = pareto_k[np.isfinite(pareto_k)]
    good_k = _float_or_none(getattr(loo_result, "good_k", None))
    warning = bool(getattr(loo_result, "warning", False))

    n_obs = int(pareto_k.size)
    n_over_good_k = int(np.sum(finite_pareto > good_k)) if good_k is not None else 0
    n_over_0_7 = int(np.sum(finite_pareto > 0.7))
    n_over_1_0 = int(np.sum(finite_pareto > 1.0))

    return {
        "warning": warning,
        "good_k": good_k,
        "pareto_k_max": float(finite_pareto.max()) if finite_pareto.size else None,
        "n_over_good_k": n_over_good_k,
        "n_over_0_7": n_over_0_7,
        "n_over_1_0": n_over_1_0,
        "n_obs": n_obs,
        "n_draws": int(n_draws) if n_draws is not None else None,
        "elpd_se": _float_or_none(
            getattr(loo_result, "se", getattr(loo_result, "elpd_loo_se", None))
        ),
    }


def _idata_num_draws(idata: Any) -> int | None:
    posterior = getattr(idata, "posterior", None)
    sizes = getattr(posterior, "sizes", None)
    if sizes is None:
        return None
    chain = sizes.get("chain")
    draw = sizes.get("draw")
    if chain is None or draw is None:
        return None
    return int(chain) * int(draw)


def build_loo_payload(loo_result: Any, *, n_draws: int | None = None) -> dict[str, Any]:
    """Convert an ArviZ LOO result into the shared SynthStats payload."""
    return {
        "elpd": float(loo_result.elpd_loo),
        "diagnostics": _summarize_loo_result(loo_result, n_draws=n_draws),
    }


class LikelihoodJudge:
    """Reward = exp(beta * ELPD-LOO). Uses ArviZ when idata is available."""

    def __init__(
        self,
        beta: float = 0.1,
        clip_range: tuple[float, float] = LOG_REWARD_CLAMP_RANGE,
        loo_gate_mode: str = "strict",
    ):
        if loo_gate_mode not in _VALID_LOO_GATE_MODES:
            valid = ", ".join(sorted(_VALID_LOO_GATE_MODES))
            raise ValueError(f"loo_gate_mode must be one of {valid}, got {loo_gate_mode!r}")
        self.beta = beta
        self.clip_range = clip_range
        self.loo_gate_mode = loo_gate_mode

    def _compute_elpd_loo(self, idata: az.InferenceData) -> dict[str, Any]:
        try:
            import arviz as az
        except ImportError as e:
            raise ImportError(
                "ArviZ required for ELPD-LOO computation. Install with: pip install arviz"
            ) from e

        loo_result = az.loo(idata, pointwise=True, scale="log")
        return build_loo_payload(loo_result, n_draws=_idata_num_draws(idata))

    def _elpd_from_log_lik_matrix(self, log_lik_matrix: np.ndarray) -> dict[str, Any]:
        import arviz as az
        import numpy as np

        arr = np.asarray(log_lik_matrix, dtype=np.float64)
        if arr.ndim != 2 or arr.size == 0:
            raise ValueError(f"log_lik_matrix must be 2-D, got shape {arr.shape}")
        if arr.shape[0] < 2:
            raise ValueError(
                f"log_lik_matrix requires at least 2 posterior draws for LOO-CV, got {arr.shape[0]}"
            )
        # arviz expects (chain, draw, obs)
        log_lik_3d = arr[np.newaxis, :, :]
        n_draws = log_lik_3d.shape[1]
        idata = az.from_dict(
            posterior={"_dummy": np.zeros((1, n_draws))},
            log_likelihood={"y": log_lik_3d},
        )
        loo_result = az.loo(idata, var_name="y", pointwise=True, scale="log")
        return build_loo_payload(loo_result, n_draws=n_draws)

    @staticmethod
    def _strict_rejects(elpd: float, diagnostics: dict[str, Any]) -> bool:
        if not math.isfinite(elpd):
            return True
        if diagnostics.get("method") == "fixed_parameter":
            n_obs = diagnostics.get("n_obs")
            standard_error = _float_or_none(diagnostics.get("elpd_se"))
            return not (
                diagnostics.get("warning") is False
                and diagnostics.get("n_draws") == 0
                and isinstance(n_obs, int)
                and not isinstance(n_obs, bool)
                and n_obs > 0
                and standard_error is not None
                and math.isfinite(standard_error)
                and standard_error >= 0
            )
        if not isinstance(diagnostics.get("warning"), bool):
            return True
        good_k = _float_or_none(diagnostics.get("good_k"))
        n_over = diagnostics.get("n_over_good_k")
        if good_k is None or not math.isfinite(good_k):
            return True
        if isinstance(n_over, bool) or not isinstance(n_over, int) or n_over < 0:
            return True
        if diagnostics["warning"]:
            return True
        return n_over > 0

    @staticmethod
    def _score_from_precomputed(
        artifacts: dict[str, Any],
        *,
        source: str,
    ) -> tuple[float, dict[str, Any], str]:
        return (
            float(artifacts["elpd"]),
            _coerce_dict(artifacts.get("diagnostics")),
            source,
        )

    def score(self, *, task_name: str, trajectory: Trajectory, artifacts: dict[str, Any]) -> Reward:
        del task_name, trajectory

        elpd: float | None = None
        elpd_source = "default"
        diagnostics: dict[str, Any] = {}
        expected_artifact_key = score_artifact_key_from_payload(artifacts)

        if expected_artifact_key == "elpd" and "elpd" in artifacts:
            elpd, diagnostics, elpd_source = self._score_from_precomputed(
                artifacts,
                source="precomputed_runtime_primary",
            )
        elif expected_artifact_key == "log_lik_matrix" and "log_lik_matrix" in artifacts:
            try:
                payload = self._elpd_from_log_lik_matrix(artifacts["log_lik_matrix"])
                elpd = float(payload["elpd"])
                diagnostics = _coerce_dict(payload.get("diagnostics"))
                elpd_source = "log_lik_matrix_loo_runtime_primary"
            except (
                RuntimeError,
                ValueError,
                TypeError,
                ArithmeticError,
                ImportError,
                IndexError,
            ) as e:
                logger.warning("log_lik_matrix LOO failed, falling back: %s", e)
                if "elpd" in artifacts:
                    elpd, diagnostics, elpd_source = self._score_from_precomputed(
                        artifacts,
                        source="precomputed_fallback",
                    )
                else:
                    elpd_source = f"error: {e}"
        elif "log_lik_matrix" in artifacts:
            try:
                payload = self._elpd_from_log_lik_matrix(artifacts["log_lik_matrix"])
                elpd = float(payload["elpd"])
                diagnostics = _coerce_dict(payload.get("diagnostics"))
                elpd_source = "log_lik_matrix_loo"
            except (
                RuntimeError,
                ValueError,
                TypeError,
                ArithmeticError,
                ImportError,
                IndexError,
            ) as e:
                logger.warning("log_lik_matrix LOO failed, falling back: %s", e)
                if "elpd" in artifacts:
                    elpd, diagnostics, elpd_source = self._score_from_precomputed(
                        artifacts,
                        source="precomputed_fallback",
                    )
                else:
                    elpd_source = f"error: {e}"
        elif "idata" in artifacts:
            try:
                payload = self._compute_elpd_loo(artifacts["idata"])
                elpd = float(payload["elpd"])
                diagnostics = _coerce_dict(payload.get("diagnostics"))
                elpd_source = "loo_cv"
            except (RuntimeError, ValueError, ArithmeticError, IndexError) as e:
                logger.warning("LOO computation failed, falling back: %s", e)
                if "elpd" in artifacts:
                    elpd, diagnostics, elpd_source = self._score_from_precomputed(
                        artifacts,
                        source="precomputed_fallback",
                    )
                else:
                    elpd_source = f"error: {e}"
        elif "elpd" in artifacts:
            elpd, diagnostics, elpd_source = self._score_from_precomputed(
                artifacts,
                source="precomputed",
            )

        if elpd_source == "default":
            return Reward(
                total=0.0,
                components={"elpd": 0.0},
                info=annotate_reward_info(
                    {
                        "log_reward": float("-inf"),
                        "elpd_source": "default_no_evidence",
                        "diagnostics": {},
                        "loo_status": "not_computed",
                        "would_reject_strict": False,
                    },
                    source="validation.likelihood",
                ),
            )

        if elpd is None:
            return Reward(
                total=0.0,
                components={},
                info=annotate_reward_info(
                    {
                        "log_reward": float("-inf"),
                        "elpd_source": elpd_source,
                        "diagnostics": diagnostics,
                        "loo_status": "computation_failed",
                        "would_reject_strict": True,
                    },
                    source="validation.likelihood",
                ),
            )

        strict_reject = self._strict_rejects(elpd, diagnostics)

        if not math.isfinite(elpd):
            return Reward(
                total=0.0,
                components={},
                info=annotate_reward_info(
                    {
                        "log_reward": float("-inf"),
                        "elpd_source": elpd_source,
                        "diagnostics": diagnostics,
                        "loo_status": "rejected_nonfinite",
                        "would_reject_strict": True,
                    },
                    source="validation.likelihood",
                ),
            )

        if self.loo_gate_mode == "strict" and strict_reject:
            return Reward(
                total=0.0,
                components={"elpd": elpd},
                info=annotate_reward_info(
                    {
                        "log_reward": float("-inf"),
                        "elpd_source": elpd_source,
                        "diagnostics": diagnostics,
                        "loo_status": "rejected_strict",
                        "would_reject_strict": True,
                    },
                    source="validation.likelihood",
                ),
            )

        log_reward = self.beta * elpd
        log_reward = max(self.clip_range[0], min(self.clip_range[1], log_reward))
        total = math.exp(log_reward)
        loo_status = "would_reject_strict" if strict_reject else "accepted"

        return Reward(
            total=total,
            components={"elpd": elpd},
            info=annotate_reward_info(
                {
                    "log_reward": log_reward,
                    "elpd_source": elpd_source,
                    "diagnostics": diagnostics,
                    "loo_status": loo_status,
                    "would_reject_strict": strict_reject,
                },
                source="validation.likelihood",
            ),
        )
