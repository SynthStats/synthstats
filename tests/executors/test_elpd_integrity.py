"""Submitted output and module changes cannot supply evaluator scores."""

import pytest

from synthstats.executors.elpd import ELPDRunner
from synthstats.executors.worker_pool import PyMCWorkerPool

MODEL = """
import pymc as pm
import numpy as np
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0.0, sigma=5.0)
    pm.Normal("obs", mu=mu, sigma=[15., 10., 16., 11.], observed=[28., 8., -3., 7.])
    idata = pm.sample(tune=20, progressbar=False)
"""


@pytest.mark.parametrize("ending", ["", "\nraise ValueError('failed')"])
def test_printed_score_without_model_is_rejected(ending):
    result = ELPDRunner(timeout=30).execute("print('__ELPD__:2.0')" + ending)
    assert "elpd" not in result
    assert "error" in result


@pytest.mark.slow
def test_evaluator_score_is_independent_of_submitted_output():
    runner = ELPDRunner(timeout=90, random_seed=0)
    honest = runner.execute(MODEL)
    attacked = runner.execute("print('__ELPD__:2.0')\n" + MODEL)
    assert "error" not in honest
    assert attacked == honest


def test_replacing_evaluator_dependency_is_rejected_before_execution():
    program = """import arviz as az
import numpy as np
class ForgedLOO:
    elpd_loo = 2.0
    pareto_k = np.zeros(4)
    good_k = 0.7
    warning = False
    se = 0.0
az.loo = lambda *args, **kwargs: ForgedLOO()
"""
    result = ELPDRunner(timeout=30, random_seed=0).execute(program + MODEL)
    assert result == {"error": "AST safety check failed"}


def test_pool_rejects_printed_score_without_model():
    with PyMCWorkerPool(pool_size=1) as pool:
        result = ELPDRunner(timeout=30, pool=pool).execute("print('__ELPD__:2.0')")
    assert "elpd" not in result
    assert "error" in result


def test_replacing_model_type_does_not_admit_nonmodel_data():
    program = """
import pymc as pm
import numpy as np
import arviz as az
pm.Model = int
model = 1
idata = az.from_dict(
    posterior={"x": np.random.normal(size=(2, 200))},
    log_likelihood={"y": np.random.normal(-1, .1, size=(2, 200, 4))},
)
"""
    result = ELPDRunner(timeout=30, random_seed=0).execute(program)
    assert "elpd" not in result
    assert "error" in result
