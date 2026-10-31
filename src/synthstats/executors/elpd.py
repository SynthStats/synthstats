"""Fit submitted models in isolation; score inference data in the supervisor."""

from __future__ import annotations

import json
import logging
import math
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from synthstats.core.constants import PYMC_EXECUTOR_TIMEOUT_S
from synthstats.executors.isolated_python import run_isolated_python
from synthstats.executors.pymc_sandbox import check_code_safety, rewrite_pm_sample_calls
from synthstats.judges.likelihood import build_loo_payload

if TYPE_CHECKING:
    from synthstats.executors.worker_pool import PyMCWorkerPool

logger = logging.getLogger(__name__)

_INFERENCE_TEMPLATE = """
import contextlib as _contextlib
import json as _json
import random as _rng
import sys as _sys
import numpy as _np
import pymc as _pm
from pymc.distributions.distribution import SymbolicRandomVariable as _SymbolicRV
from pytensor import function as _function
from pytensor.graph.traversal import ancestors as _ancestors
from pytensor.tensor.random.op import RandomVariable as _RandomRV

_model_type = _pm.Model
_type = type
_logp = _pm.logp
_sample = _pm.sample
_compute_log_likelihood = _pm.compute_log_likelihood
_asarray = _np.asarray
_dumps = _json.dumps
_scope = {"np": _np, "pm": _pm}
# The child returns inference data, never a score. Keep submitted print output
# out of the data transport, including prints from model-building callbacks.
with _contextlib.redirect_stdout(_sys.stderr):
    __SEED_PREAMBLE__
    exec(__PROGRAM__, _scope)
    _model = _scope.get("model")
    if _type(_model) is not _model_type:
        _models = [value for value in _scope.values() if _type(value) is _model_type]
        if not _models:
            raise NameError("model is not defined")
        _model = _models[-1]
    # Submitted inference objects are untrusted, even when a real model exists.
    # Only evaluator-owned sampling and likelihood evaluation feed the transport.
    with _model:
        if not _model.free_RVs:
            # Build each observed density without replacing its random inputs
            # from the submitted model's mutable RV-to-value mapping.
            _observed_logps = [
                _logp(_rv, _model.rvs_to_values[_rv]) for _rv in _model.observed_RVs
            ]
            if any(
                _var.owner is not None
                and isinstance(_var.owner.op, (_RandomRV, _SymbolicRV))
                for _var in _ancestors(_observed_logps)
            ) or _model.rvs_to_initial_values:
                raise ValueError("fixed-parameter model contains free random variables")
            _values = _function([], _observed_logps, accept_inplace=True)()
            _groups = {"posterior": {}, "log_likelihood": {
                _rv.name: _asarray(_value)[None, None, ...].tolist()
                for _rv, _value in zip(_model.observed_RVs, _values)
            }}
        else:
            _idata = _sample(
                draws=200, chains=2, cores=1, progressbar=False,
                compute_convergence_checks=False, return_inferencedata=True,
                idata_kwargs={"log_likelihood": False}__SEED_KW__
            )
            _idata = _compute_log_likelihood(_idata, model=_model)
            _groups = {}
            for _name in ("posterior", "log_likelihood"):
                _group = getattr(_idata, _name)
                _groups[_name] = {
                    _key: _asarray(_value.transpose("chain", "draw", ...)).tolist()
                    for _key, _value in _group.data_vars.items()
                }
print(_dumps(_groups, allow_nan=False))
"""


def _score_inference_output(stdout: str, stderr: str, returncode: int) -> dict[str, Any]:
    """Accept numeric inference data only, then run evaluator-owned ArviZ.

    Nothing deserialized here can execute Python or replace a scoring function.
    Submitted stdout markers and precomputed scores are not a result protocol.
    """
    if returncode != 0:
        logger.warning("Program failed: %s", stderr[:300])
        return {"error": "model evaluation failed"}

    import arviz as az
    import numpy as np

    try:
        raw = json.loads(stdout)
        if not isinstance(raw, dict) or set(raw) != {"posterior", "log_likelihood"}:
            raise ValueError("expected posterior and log_likelihood groups")
        groups = {}
        sample_shape = None
        for name in ("posterior", "log_likelihood"):
            group = raw[name]
            if not isinstance(group, dict) or (name == "log_likelihood" and not group):
                raise ValueError(f"missing {name} data")
            arrays = {}
            for key, values in group.items():
                array = np.asarray(values)
                if array.dtype.kind not in "biuf" or array.ndim < 2:
                    raise ValueError("inference variables must be numeric chain/draw arrays")
                if array.size == 0 or not np.isfinite(array).all():
                    raise ValueError("inference variables must be nonempty and finite")
                if sample_shape is None:
                    sample_shape = array.shape[:2]
                if array.shape[:2] != sample_shape:
                    raise ValueError("inconsistent posterior sample dimensions")
                arrays[key] = array
            groups[name] = arrays
        # Each scalar likelihood factor is one held-out observation unit. PyMC
        # has already reduced a multivariate event to its joint log density.
        # Distinct observed variables contribute disjoint sets of those units.
        assert sample_shape is not None
        combined_likelihood = np.concatenate(
            [array.reshape(*sample_shape, -1) for array in groups["log_likelihood"].values()],
            axis=-1,
        )
        if not groups["posterior"]:
            if sample_shape != (1, 1):
                raise ValueError("fixed models require one exact likelihood evaluation")
            # With no unknown parameters, omitting an observation cannot change
            # the predictive distribution. LOO is its exact log density; PSIS
            # and posterior-sampling diagnostics do not apply.
            pointwise = combined_likelihood[0, 0]
            with np.errstate(over="raise", invalid="raise"):
                exact_elpd = float(pointwise.sum())
                exact_se = float(np.sqrt(pointwise.size * pointwise.var()))
            return {
                "elpd": exact_elpd,
                "diagnostics": {
                    "method": "fixed_parameter",
                    "warning": False,
                    "good_k": None,
                    "pareto_k_max": None,
                    "n_over_good_k": None,
                    "n_over_0_7": None,
                    "n_over_1_0": None,
                    "n_obs": int(pointwise.size),
                    "n_draws": 0,
                    "elpd_se": exact_se,
                },
            }
        if np.prod(sample_shape) < 2:
            raise ValueError("inconsistent posterior sample dimensions")
        idata = az.from_dict(
            posterior=groups["posterior"], log_likelihood={"observations": combined_likelihood}
        )
        loo = az.loo(idata, pointwise=True, scale="log")
        pareto_k = np.asarray(loo.pareto_k)
        if pareto_k.size == 0 or not np.isfinite(pareto_k).all():
            raise ValueError("missing or nonfinite LOO diagnostics")
        payload = build_loo_payload(loo, n_draws=math.prod(sample_shape))
        diagnostics = payload["diagnostics"]
        if not all(
            isinstance(value, int | float) and math.isfinite(value)
            for value in (payload["elpd"], diagnostics["good_k"], diagnostics["elpd_se"])
        ):
            raise ValueError("incomplete LOO result")
        return payload
    except (ValueError, TypeError, KeyError, AttributeError, ArithmeticError, IndexError) as exc:
        logger.warning("Inference scoring failed: %s", exc)
        return {"error": f"inference scoring failed: {exc}"}


@dataclass
class ELPDRunner:
    """Fit models in isolated children and compute ELPD in the supervisor.

    A pool supervises fresh isolated children in the same way as one-shot runs.
    Only numeric posterior and log-likelihood arrays cross into the scorer.
    """

    timeout: float = PYMC_EXECUTOR_TIMEOUT_S
    random_seed: int | None = None
    pool: PyMCWorkerPool | None = None

    def execute(self, program: str) -> dict[str, Any]:
        prepared = self._prepare_program(program)
        if prepared is None:
            return {"error": "AST safety check failed"}
        if self.pool is not None:
            return self._run_via_pool(prepared)
        try:
            result = run_isolated_python(prepared, timeout=self.timeout)
        except (RuntimeError, subprocess.SubprocessError, OSError) as exc:
            logger.warning("Program exec error: %s", exc)
            return {"error": "subprocess failed or timed out"}
        return _score_inference_output(result.stdout, result.stderr, result.returncode)

    def _run_via_pool(self, code: str) -> dict[str, Any]:
        from synthstats.executors.worker_pool import WorkerError

        try:
            result = self.pool.submit(  # type: ignore[union-attr]
                code, seed=self.random_seed, timeout=self.timeout
            )
        except WorkerError as exc:
            logger.warning("Worker pool error: %s", exc)
            return {"error": f"worker pool error: {exc}"}
        return _score_inference_output(
            result.get("stdout", ""), result.get("stderr", ""), result.get("returncode", 1)
        )

    def _prepare_program(self, program: str) -> str | None:
        is_safe, error = check_code_safety(program)
        if not is_safe:
            logger.warning("AST safety check failed: %s", error)
            return None
        modified = rewrite_pm_sample_calls(program, random_seed=self.random_seed)
        seed_preamble = "pass"
        seed_kw = ""
        if self.random_seed is not None:
            seed_preamble = f"_np.random.seed({self.random_seed}); _rng.seed({self.random_seed})"
            seed_kw = f", random_seed={self.random_seed}"
        return (
            _INFERENCE_TEMPLATE.replace("__SEED_PREAMBLE__", seed_preamble)
            .replace("__SEED_KW__", seed_kw)
            .replace("__PROGRAM__", repr(modified))
        )


__all__ = ["ELPDRunner"]
