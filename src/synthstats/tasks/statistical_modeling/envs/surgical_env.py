"""Surgical synthetic environment.

True model: deaths_j ~ Poisson(exposure_j * hazard_j),
log(hazard_j) ~ N(mu, sigma). Agent queries hospital IDs for (deaths, exposure).
"""

from __future__ import annotations

import math
import random

from synthstats.tasks.statistical_modeling.envs.query_parsing import parse_query_int

_LAZYPPL_COLUMNS = [
    {"name": "deaths", "hs_type": "Int"},
    {"name": "exposure", "hs_type": "Int"},
]


class SurgicalEnv:
    # exposures (procedure counts), synthetic
    DEFAULT_EXPOSURES: tuple[int, ...] = (
        1767,
        2367,
        2145,
        1098,
        2389,
        1876,
        2134,
        1567,
        2897,
        1234,
        1876,
        2456,
    )

    def __init__(self):
        self.n_hospitals = 12
        self.mu = -5.0  # baseline log-hazard (low mortality)
        self.sigma = 0.5  # between-hospital variation
        self.log_hazards: list[float] = []
        self.hazards: list[float] = []
        self.deaths: list[int] = []
        self.exposures: list[int] = list(self.DEFAULT_EXPOSURES)
        self.rng = random.Random()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng.seed(seed)

        self.mu = self.rng.uniform(-6.0, -4.0)
        self.sigma = self.rng.uniform(0.3, 0.8)

        self.log_hazards = [self.rng.gauss(self.mu, self.sigma) for _ in range(self.n_hospitals)]
        self.hazards = [math.exp(lh) for lh in self.log_hazards]

        self.exposures = [
            max(100, int(e * self.rng.uniform(0.8, 1.2))) for e in self.DEFAULT_EXPOSURES
        ]

        self.deaths = [
            self._poisson(self.exposures[j] * self.hazards[j]) for j in range(self.n_hospitals)
        ]

    def _poisson(self, lam: float) -> int:
        # Knuth algorithm for small lambda, normal approximation otherwise
        if lam < 30:
            L = math.exp(-lam)
            k = 0
            p = 1.0
            while p > L:
                k += 1
                p *= self.rng.random()
            return k - 1
        else:
            return max(0, round(self.rng.gauss(lam, math.sqrt(lam))))

    def query(self, query: str) -> str:
        hospital_id = parse_query_int(query, "hospital")
        if hospital_id is None:
            return f"Invalid query. Use format: hospital=<1-{self.n_hospitals}>"
        if not 1 <= hospital_id <= self.n_hospitals:
            return f"Invalid hospital ID. Use 1-{self.n_hospitals}."

        idx = hospital_id - 1
        deaths = self.deaths[idx]
        exposure = self.exposures[idx]
        return f"deaths={deaths}, exposure={exposure}"

    @staticmethod
    def get_lazyppl_columns() -> list[dict[str, str]]:
        return list(_LAZYPPL_COLUMNS)

    @staticmethod
    def get_lazyppl_prompt_info() -> dict[str, str]:
        return {
            "query_example": "hospital=1",
            "query_description": "Query with hospital=<1-12> to observe deaths and exposure",
            "row_fields_doc": (
                "`deaths r` (Int, observed deaths) and `exposure r` (Int, number of procedures)"
            ),
        }

    @staticmethod
    def get_lazyppl_eval_data(seed: int = 7) -> list[dict]:
        """All 12 hospitals using default exposures and fixed-seed deaths."""
        mu, sigma = -5.0, 0.5
        rng = random.Random(seed)
        exposures = list(SurgicalEnv.DEFAULT_EXPOSURES)

        rows = []
        for j in range(12):
            log_hazard = rng.gauss(mu, sigma)
            hazard = math.exp(log_hazard)
            lam = exposures[j] * hazard

            # Knuth algorithm for small lambda
            if lam < 30:
                L = math.exp(-lam)
                k = 0
                p = 1.0
                while p > L:
                    k += 1
                    p *= rng.random()
                deaths = k - 1
            else:
                deaths = max(0, round(rng.gauss(lam, math.sqrt(lam))))

            rows.append({"deaths": deaths, "exposure": exposures[j]})
        return rows
