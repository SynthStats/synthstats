"""Observation factors retain their scores across likelihood-variable partitions."""

import json

import arviz as az
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from synthstats.executors.elpd import ELPDRunner, _score_inference_output


@given(left=st.integers(1, 4), right=st.integers(1, 4))
def test_disjoint_likelihood_variables_score_every_observation(left, right):
    rng = np.random.default_rng(41)
    posterior = {"mu": rng.normal(size=(2, 200))}
    likelihood = rng.normal(-1.0, 0.1, size=(2, 200, left + right))
    reference = az.loo(
        az.from_dict(posterior=posterior, log_likelihood={"obs": likelihood}),
        pointwise=True,
        scale="log",
    )
    encoded = json.dumps(
        {
            "posterior": {name: value.tolist() for name, value in posterior.items()},
            "log_likelihood": {
                "obs_a": likelihood[:, :, :left].tolist(),
                "obs_b": likelihood[:, :, left:].tolist(),
            },
        }
    )

    result = _score_inference_output(encoded, "", 0)

    assert "error" not in result, result
    assert result["elpd"] == pytest.approx(float(reference.elpd_loo), abs=1e-12)
    assert result["diagnostics"]["n_obs"] == left + right


@pytest.mark.slow
def test_elpd_runner_scores_a_model_with_disjoint_observed_variables():
    program = """
import pymc as pm
with pm.Model() as model:
    mu = pm.Normal("mu", 0, 1)
    pm.Normal("obs_a", mu, 1, observed=[0., 0.5])
    pm.Normal("obs_b", mu, 1, observed=[1., 1.5])
"""
    result = ELPDRunner(timeout=90, random_seed=0).execute(program)

    assert "error" not in result, result
    assert np.isfinite(result["elpd"])
    assert result["diagnostics"]["n_obs"] == 4
