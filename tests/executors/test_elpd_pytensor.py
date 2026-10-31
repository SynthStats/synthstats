"""PyTensor tensor expressions are part of the PyMC model language."""

import math

import pytest

from synthstats.executors.elpd import ELPDRunner


@pytest.mark.slow
@pytest.mark.parametrize(
    "import_statement",
    [
        "import pytensor.tensor as pt",
        "from pytensor import tensor as pt",
        "from pytensor.tensor import stack",
    ],
)
def test_elpd_runner_scores_pytensor_model_building(import_statement):
    stack = "stack" if import_statement.endswith("import stack") else "pt.stack"
    program = f"""
import pymc as pm
{import_statement}
with pm.Model() as model:
    mu = pm.Normal("mu", 0, 1)
    means = {stack}([mu, mu])
    pm.Normal("obs", means, 1, observed=[0., 1.])
"""
    result = ELPDRunner(timeout=90, random_seed=0).execute(program)

    assert "error" not in result, result
    assert math.isfinite(result["elpd"])
    assert result["diagnostics"]["n_obs"] == 2
