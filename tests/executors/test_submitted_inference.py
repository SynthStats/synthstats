"""The verifier samples the model instead of trusting submitted inference data."""

import pytest

from synthstats.executors.elpd import ELPDRunner

MODEL = """
import numpy as np
import pymc as pm
import arviz as az
with pm.Model() as model:
    mu = pm.Normal('mu', mu=0.0, sigma=5.0)
    pm.Normal('obs', mu=mu, sigma=[15., 10., 16., 11.], observed=[28., 8., -3., 7.])
"""


@pytest.mark.slow
@pytest.mark.parametrize("shift", [0, 10])
def test_submitted_log_likelihood_cannot_change_model_score(shift):
    program = (
        MODEL
        + f"""
idata = az.from_dict(
    posterior={{'mu': np.random.normal(size=(2, 200))}},
    log_likelihood={{'obs': np.random.normal(-1, .1, size=(2, 200, 4)) + {shift}}},
)
"""
    )
    result = ELPDRunner(timeout=90, random_seed=0).execute(program)
    assert "error" not in result
    # The same score as the model alone: the submitted log-likelihood is ignored.
    assert result["elpd"] == pytest.approx(-15.691536124801045, rel=0, abs=1e-12)
    assert result["diagnostics"]["n_obs"] == 4


@pytest.mark.slow
def test_submitted_posterior_without_log_likelihood_is_ignored():
    program = (
        MODEL
        + """
idata = az.from_dict(posterior={'mu': np.full((2, 200), 1000.)})
trace = posterior = log_likelihood = np.zeros((2, 200, 4))
"""
    )
    result = ELPDRunner(timeout=90, random_seed=0).execute(program)
    assert "error" not in result
    assert result["elpd"] == pytest.approx(-15.691536124801045, rel=0, abs=1e-12)
    assert result["diagnostics"]["n_obs"] == 4
