"""Submitted programs cannot mutate the libraries used by the evaluator."""

import pytest

from synthstats.core.types import ToolCall
from synthstats.executors.elpd import ELPDRunner
from synthstats.executors.pymc_sandbox import PyMCExecutor, check_code_safety


@pytest.mark.parametrize(
    "program",
    [
        "import pymc as pm; pm.Model = int",
        "import numpy as np; np.random.x = 1",
        "import pymc.sampling.mcmc as sampling; sampling.sample = int",
        "from pymc import sampling; sampling.mcmc.sample = int",
        "import numpy as np; library = np.random; library.normal = int",
        "import numpy as np; libraries = [np]; libraries[0].asarray = int",
        "pm.compute_log_likelihood = int",
        "import numpy as np; np.pi += 1",
        "import numpy as np; np.pi: float = 0",
        "import numpy as np; del np.asarray",
        "import numpy as np\nfor np.pi in []: pass",
    ],
)
def test_rejects_attribute_mutation(program):
    safe, reason = check_code_safety(program)
    assert not safe
    assert reason


@pytest.mark.parametrize(
    "program",
    [
        "import numpy as np; np.random.mtrand.normal = lambda *a, **k: 0",
        "import pymc.sampling.mcmc as sampling; sampling.sample = int",
    ],
)
def test_library_tampering_is_rejected_before_execution(program):
    # If executed, this program raises before it can patch a library or fit a model.
    program = "raise RuntimeError('submission executed')\n" + program
    assert ELPDRunner(timeout=30).execute(program) == {"error": "AST safety check failed"}
    result = PyMCExecutor().execute(
        ToolCall(name="pymc", raw="", input={"code": program}), timeout_s=30
    )
    assert not result.success
    assert result.output == ""
    assert result.error and "Blocked" in result.error


@pytest.mark.parametrize("attribute", ["__dict__", "__class__", "__name__", "__doc__"])
def test_rejects_every_dunder_attribute(attribute):
    safe, reason = check_code_safety(f"import numpy as np; value = np.{attribute}")
    assert not safe
    assert reason


@pytest.mark.parametrize(
    "program",
    [
        "import json",
        "from collections import namedtuple",
        "from . import hidden",
        "from .numpy import array",
        "from numpy import *",
        "from numpy import __dict__ as namespace",
        "from numpy import __name__ as name",
    ],
)
def test_rejects_imports_outside_explicit_allowlist(program):
    safe, reason = check_code_safety(program)
    assert not safe
    assert reason


@pytest.mark.parametrize(
    "program",
    [
        "import pymc as pm; setattr(pm, 'Model', int)",
        "import pymc as pm; delattr(pm, 'Model')",
        "exec('pass')",
        "eval('1')",
        "compile('pass', '<program>', 'exec')",
        "globals()",
        "vars()",
        "import pymc as pm; getattr(pm, 'Model')",
        "function = setattr",
        "function = eval",
        "namespace = __builtins__",
    ],
)
def test_rejects_dynamic_execution_and_reflection(program):
    safe, reason = check_code_safety(program)
    assert not safe
    assert reason


@pytest.mark.parametrize(
    "program",
    [
        "import numpy as np; values = np.array([1, 2]); values[0] = 3",
        "from math import sqrt as root; result = root(4)",
        "from pymc import Normal; mu = Normal('mu', mu=0, sigma=1)",
        "import pymc.sampling.mcmc as sampling; sample = sampling.sample",
    ],
)
def test_allows_allowlisted_imports_and_local_array_updates(program):
    assert check_code_safety(program) == (True, None)
