"""Eight Schools synthetic environment.

Hierarchical model: theta_j ~ N(mu, tau), y_j ~ N(theta_j, sigma_j)
where sigma_j are known standard errors from Rubin (1981).
"""

from __future__ import annotations

import random

from synthstats.tasks.statistical_modeling.envs.query_parsing import parse_query_int

_LAZYPPL_COLUMNS = [
    {"name": "effect", "hs_type": "Double"},
    {"name": "se", "hs_type": "Double"},
]


class EightSchoolsEnv:
    # known standard errors from Rubin (1981)
    KNOWN_SES: tuple[float, ...] = (15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0)

    def __init__(self):
        self.mu = 0.0  # global mean
        self.tau = 5.0  # between-school sd
        self.theta = [0.0] * 8  # true effects
        self.observed_effects = [0.0] * 8  # observed with noise
        self.rng = random.Random()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng.seed(seed)

        self.mu = self.rng.gauss(0, 10)
        self.tau = abs(self.rng.gauss(0, 10))

        # floor tau at 0.1 to avoid degenerate sampling
        self.theta = [self.rng.gauss(self.mu, max(0.1, self.tau)) for _ in range(8)]

        self.observed_effects = [self.rng.gauss(self.theta[j], self.KNOWN_SES[j]) for j in range(8)]

    def query(self, query: str) -> str:
        school_id = parse_query_int(query, "school")
        if school_id is None:
            return "Invalid query. Use format: school=<1-8>"
        if not 1 <= school_id <= 8:
            return "Invalid school ID. Use 1-8."

        idx = school_id - 1
        effect = self.observed_effects[idx]
        se = self.KNOWN_SES[idx]
        return f"effect={effect:.1f}, se={se:.1f}"

    @staticmethod
    def get_lazyppl_columns() -> list[dict[str, str]]:
        return list(_LAZYPPL_COLUMNS)

    @staticmethod
    def get_lazyppl_prompt_info() -> dict[str, str]:
        return {
            "query_example": "school=1",
            "query_description": (
                "Query with school=<1-8> to observe treatment effect and standard error"
            ),
            "row_fields_doc": (
                "`effect r` (Double, observed effect) and `se r` (Double, known standard error)"
            ),
        }

    @staticmethod
    def get_lazyppl_eval_data(seed: int = 7) -> list[dict]:
        """All 8 schools with known SEs and fixed-seed observed effects."""
        rng = random.Random(seed)
        mu = rng.gauss(0, 10)
        tau = abs(rng.gauss(0, 10))
        theta = [rng.gauss(mu, max(0.1, tau)) for _ in range(8)]
        rows = []
        for j in range(8):
            se = EightSchoolsEnv.KNOWN_SES[j]
            effect = rng.gauss(theta[j], se)
            rows.append({"effect": round(effect, 2), "se": se})
        return rows
