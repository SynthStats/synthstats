import sys
from types import ModuleType

import pytest

from synthstats.tasks.statistical_modeling.task import StatisticalModelingTask


@pytest.mark.parametrize("name", ["dugongs", "eight_schools", "peregrines", "surgical"])
def test_installed_optional_package_cannot_replace_the_packaged_task(monkeypatch, name):
    class UnexpectedEnvironment:
        def __init__(self):
            raise AssertionError("An installed package replaced the packaged task")

    registry = ModuleType("boxing_gym.envs.registry")
    registry.get_environment_registry = lambda: ({name: UnexpectedEnvironment}, {})
    for module_name in ("boxing_gym", "boxing_gym.envs"):
        monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    task = StatisticalModelingTask(env_name=name)
    task.env.reset(seed=17)
    assert task.env.__class__.__module__.startswith("synthstats.tasks.statistical_modeling.envs.")
