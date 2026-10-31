from __future__ import annotations

import pytest

from synthstats.executors.elpd import ELPDRunner
from synthstats.executors.pymc_sandbox import rewrite_pm_sample_calls


@pytest.mark.slow
def test_elpd_runner_executes_model_only_program_without_explicit_idata() -> None:
    program = """
import pymc as pm
import numpy as np

observed = np.array([28.0, 8.0, -3.0, 7.0])
sigma = np.array([15.0, 10.0, 16.0, 11.0])

with pm.Model() as location_model:
    mu = pm.Normal("mu", mu=0.0, sigma=5.0)
    pm.Normal("obs", mu=mu, sigma=sigma, observed=observed)
"""

    result = ELPDRunner(timeout=60.0, random_seed=0).execute(program)

    assert "error" not in result
    assert isinstance(result.get("elpd"), float)
    # The seeded fixture has a fixed reference score.
    assert result["elpd"] == pytest.approx(-15.691536124801045, rel=0, abs=1e-12)
    diagnostics = result.get("diagnostics")
    assert isinstance(diagnostics, dict)
    assert diagnostics.get("n_obs") == 4
    assert diagnostics.get("n_draws") == 400


def test_elpd_runner_prepare_program_has_no_repo_import_dependency() -> None:
    prepared = ELPDRunner()._prepare_program("import pymc as pm\nwith pm.Model() as m:\n    pass\n")

    assert prepared is not None
    assert "from synthstats" not in prepared


def test_rewrite_pm_sample_calls_injects_random_seed_when_requested() -> None:
    rewritten = rewrite_pm_sample_calls(
        "import pymc as pm\npm.sample(draws=50)\n",
        random_seed=123,
    )

    assert "draws=200" in rewritten
    assert "chains=2" in rewritten
    assert "cores=1" in rewritten
    assert "random_seed=123" in rewritten


@pytest.mark.slow
def test_elpd_runner_seeded_execution_is_deterministic() -> None:
    program = """
import pymc as pm
import numpy as np

observed = np.array([28.0, 8.0, -3.0, 7.0])
sigma = np.array([15.0, 10.0, 16.0, 11.0])

with pm.Model() as location_model:
    mu = pm.Normal("mu", mu=0.0, sigma=5.0)
    pm.Normal("obs", mu=mu, sigma=sigma, observed=observed)
"""

    first = ELPDRunner(timeout=60.0, random_seed=0).execute(program)
    second = ELPDRunner(timeout=60.0, random_seed=0).execute(program)

    assert "error" not in first
    assert "error" not in second
    assert first["elpd"] == second["elpd"]
    assert first["diagnostics"] == second["diagnostics"]
