"""Fixed models have exact predictive densities and require no posterior draws."""

import json
import math

import pytest

from synthstats.core.types import Reward, Trajectory
from synthstats.executors.elpd import ELPDRunner, _score_inference_output
from synthstats.judges.likelihood import LikelihoodJudge


@pytest.mark.slow
@pytest.mark.parametrize("split", [False, True])
def test_fixed_parameter_model_receives_its_exact_predictive_reward(split):
    observations = (
        'pm.Normal("a", 0, 1, observed=[0.])\n    pm.Normal("b", 0, 1, observed=[1.])'
        if split
        else 'pm.Normal("obs", 0, 1, observed=[0., 1.])'
    )
    program = f"import pymc as pm\nwith pm.Model() as model:\n    {observations}\n"

    result = ELPDRunner(timeout=90, random_seed=0).execute(program)

    assert "error" not in result, result
    # The two Normal(0, 1) log densities are -log(2*pi)/2 and that value minus 1/2.
    expected = -math.log(2 * math.pi) - 0.5
    assert result["elpd"] == pytest.approx(expected, abs=1e-12)
    diagnostics = result["diagnostics"]
    assert diagnostics["method"] == "fixed_parameter"
    assert diagnostics["n_obs"] == 2
    assert diagnostics["n_draws"] == 0
    assert diagnostics["pareto_k_max"] is None
    assert diagnostics["elpd_se"] == pytest.approx(math.sqrt(0.125))
    reward = LikelihoodJudge(beta=0.1).score(
        task_name="fixed_normal",
        trajectory=Trajectory(
            messages=[],
            token_ids=[],
            token_logprobs=[],
            loss_mask=[],
            reward=Reward(total=0.0, components={}, info={}),
        ),
        artifacts=result,
    )
    assert reward.total == pytest.approx(math.exp(0.1 * expected))


@pytest.mark.slow
@pytest.mark.parametrize("transform", ["log", "unit_jacobian"])
def test_fixed_parameter_reward_uses_observed_density_despite_transform_metadata(transform):
    program = """
import pymc as pm
with pm.Model() as model:
    obs = pm.Normal("obs", 0, 1, observed=[0., 1.])
"""
    if transform == "log":
        program += "model.rvs_to_transforms[obs] = pm.distributions.transforms.log\n"
    else:
        program += """
class UnitJacobian(pm.distributions.transforms.LogTransform):
    def forward(self, value, *inputs):
        return value
    def backward(self, value, *inputs):
        return value
    def log_jac_det(self, value, *inputs):
        return pm.math.ones_like(value)
model.rvs_to_transforms[obs] = UnitJacobian()
"""

    result = ELPDRunner(timeout=90, random_seed=0).execute(program)

    assert "error" not in result, result
    # Normal(0, 1) at [0, 1]: -log(2*pi) - 1/2, independent of transforms.
    assert result["elpd"] == pytest.approx(-2.3378770664093453, abs=1e-12)
    diagnostics = result["diagnostics"]
    assert diagnostics["method"] == "fixed_parameter"
    assert diagnostics["n_obs"] == 2
    assert diagnostics["n_draws"] == 0
    assert diagnostics["elpd_se"] == pytest.approx(math.sqrt(0.125))
    reward = LikelihoodJudge(beta=0.1).score(
        task_name="fixed_normal",
        trajectory=Trajectory(
            messages=[],
            token_ids=[],
            token_logprobs=[],
            loss_mask=[],
            reward=Reward(total=0.0, components={}, info={}),
        ),
        artifacts=result,
    )
    assert reward.total == pytest.approx(0.7915298345877118, abs=1e-12)
    assert reward.info["loo_status"] == "accepted"


@pytest.mark.slow
@pytest.mark.parametrize("inplace", [False, True])
def test_fixed_parameter_model_accepts_a_declared_inplace_subtensor(inplace):
    program = f"""
import pymc as pm
import pytensor.tensor as pt
with pm.Model() as model:
    sigmas = pt.inc_subtensor(pt.ones(2)[0], 1., inplace={inplace!r})
    pm.Normal("obs", 0., sigmas, observed=[0., 1.])
"""

    result = ELPDRunner(timeout=90, random_seed=0).execute(program)

    assert "error" not in result, result
    # sigmas is [2, 1]: log N(0; 0, 2^2) + log N(1; 0, 1) = -log(4*pi) - 1/2.
    expected = -math.log(4 * math.pi) - 0.5
    assert result["elpd"] == pytest.approx(expected, abs=1e-12)
    diagnostics = result["diagnostics"]
    assert diagnostics["method"] == "fixed_parameter"
    assert diagnostics["n_obs"] == 2
    assert diagnostics["n_draws"] == 0


def test_fixed_parameter_scoring_rejects_nonfinite_aggregate_density():
    encoded = json.dumps(
        {
            "posterior": {},
            "log_likelihood": {"obs": [[[-1e308, -1e308]]]},
        }
    )
    result = _score_inference_output(encoded, "", 0)
    assert "error" in result
    assert "elpd" not in result


@pytest.mark.slow
@pytest.mark.parametrize("clear_initial_values", [False, True])
def test_mutated_free_variable_registry_cannot_receive_a_fixed_parameter_score(
    clear_initial_values,
):
    program = """
import pymc as pm
with pm.Model() as model:
    mu = pm.Normal("mu", 0, 1)
    pm.Normal("obs", mu, 1, observed=[0., 1.])
model.free_RVs.clear()
model.rvs_to_values[mu] = pm.math.constant(0.5, dtype=mu.dtype)
"""
    if clear_initial_values:
        program += "model.rvs_to_initial_values.clear()\n"

    result = ELPDRunner(timeout=90, random_seed=0).execute(program)

    assert "error" in result, result
    assert "elpd" not in result


@pytest.mark.slow
def test_model_subclass_cannot_supply_its_own_fixed_parameter_log_density():
    program = """
import pymc as pm
class SuppliedLogp(pm.Model):
    def compile_logp(self, *args, **kwargs):
        return lambda point: [[100.0, 100.0]]
with SuppliedLogp() as model:
    pm.Normal("obs", 0, 1, observed=[0., 1.])
"""

    result = ELPDRunner(timeout=90, random_seed=0).execute(program)

    assert "error" in result, result
    assert "elpd" not in result


@pytest.mark.slow
@pytest.mark.parametrize("component_list", [False, True])
def test_fixed_mixture_with_marginalized_components_receives_an_exact_score(component_list):
    components = (
        "[pm.Normal.dist(-1., 1.), pm.Normal.dist(1., 1.)]"
        if component_list
        else "pm.Normal.dist(mu=[-1., 1.], sigma=1.)"
    )
    program = f"""
import pymc as pm
with pm.Model() as model:
    pm.Mixture("obs", w=[0.5, 0.5],
               comp_dists={components}, observed=[0., 1.])
"""

    result = ELPDRunner(timeout=90, random_seed=0).execute(program)

    assert "error" not in result, result
    # Equal Normal(-1, 1) / Normal(1, 1) mixture, evaluated at 0 and 1.
    expected = -math.log(2 * math.pi) - 0.5 + math.log((1 + math.exp(-2)) / 2)
    assert result["elpd"] == pytest.approx(expected, abs=1e-12)
    assert result["diagnostics"]["method"] == "fixed_parameter"
