"""Peregrines environment.

Discrete logistic growth: N_t = N_{t-1} + r * N_{t-1} * (1 - N_{t-1}/K)
with Gaussian observation noise. Agent queries years to observe counts.
"""

from __future__ import annotations

import random

from synthstats.tasks.statistical_modeling.envs.query_parsing import parse_query_int

_LAZYPPL_COLUMNS = [
    {"name": "year", "hs_type": "Int"},
    {"name": "population", "hs_type": "Int"},
]

_EVAL_YEARS = [0, 1, 2, 3, 5, 8, 10, 15, 20, 30]


class PeregrinesEnv:
    def __init__(self):
        self.r = 2.0  # growth rate
        self.K = 200.0  # carrying capacity
        self.N_0 = 50.0  # initial population
        self.noise_std = 5.0
        self.rng = random.Random()
        self._population_cache: dict[int, float] = {}

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng.seed(seed)

        self.r = self.rng.uniform(1.5, 2.5)
        self.K = self.rng.uniform(100, 500)
        self.N_0 = self.rng.uniform(20, 80)
        self.noise_std = self.rng.uniform(3.0, 8.0)
        self._population_cache = {0: self.N_0}

    def _get_population(self, year: int) -> float:
        if year in self._population_cache:
            return self._population_cache[year]

        if year < 0:
            return self.N_0

        max_cached = max(self._population_cache.keys())
        N = self._population_cache[max_cached]

        for t in range(max_cached + 1, year + 1):
            N = N + self.r * N * (1 - N / self.K)
            N = max(0, N)  # population can't be negative
            self._population_cache[t] = N

        return N

    def query(self, query: str) -> str:
        year = parse_query_int(query, "year")
        if year is None:
            return "Invalid query. Use format: year=<integer>"
        if year < 0:
            return "Invalid query. Year must be non-negative."

        true_pop = self._get_population(year)
        noise = self.rng.gauss(0, self.noise_std)
        observed = max(0, round(true_pop + noise))
        return f"population={observed}"

    @staticmethod
    def get_lazyppl_columns() -> list[dict[str, str]]:
        return list(_LAZYPPL_COLUMNS)

    @staticmethod
    def get_lazyppl_prompt_info() -> dict[str, str]:
        return {
            "query_example": "year=5",
            "query_description": "Query with year=<integer> to observe population count",
            "row_fields_doc": "`year r` (Int) and `population r` (Int, observed count)",
        }

    @staticmethod
    def get_lazyppl_eval_data(seed: int = 7) -> list[dict]:
        """10 observations at fixed years using default parameters."""
        r, K, N_0, noise_std = 2.0, 200.0, 50.0, 5.0
        rng = random.Random(seed)

        # simulate the logistic growth trajectory
        pop_cache: dict[int, float] = {0: N_0}
        max_year = max(_EVAL_YEARS)
        N = N_0
        for t in range(1, max_year + 1):
            N = N + r * N * (1 - N / K)
            N = max(0.0, N)
            pop_cache[t] = N

        rows = []
        for yr in _EVAL_YEARS:
            observed = max(0, round(pop_cache[yr] + rng.gauss(0, noise_std)))
            rows.append({"year": yr, "population": observed})
        return rows
