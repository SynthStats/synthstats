"""Statistical modelling environment adapters and built-in envs."""

from synthstats.tasks.statistical_modeling.envs.dugongs_env import DugongsEnv
from synthstats.tasks.statistical_modeling.envs.eight_schools_env import (
    EightSchoolsEnv,
)
from synthstats.tasks.statistical_modeling.envs.peregrines_env import PeregrinesEnv
from synthstats.tasks.statistical_modeling.envs.query_parsing import (
    parse_query_float,
    parse_query_int,
    parse_query_value,
)
from synthstats.tasks.statistical_modeling.envs.surgical_env import SurgicalEnv

__all__ = [
    "DugongsEnv",
    "EightSchoolsEnv",
    "PeregrinesEnv",
    "SurgicalEnv",
    "parse_query_float",
    "parse_query_int",
    "parse_query_value",
]
