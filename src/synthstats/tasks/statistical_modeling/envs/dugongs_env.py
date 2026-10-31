"""Dugongs nonlinear growth environment."""

from __future__ import annotations

import random

from synthstats.tasks.statistical_modeling.envs.query_parsing import parse_query_float

_LAZYPPL_COLUMNS = [
    {"name": "xVal", "hs_type": "Double"},
    {"name": "yVal", "hs_type": "Double"},
]

_EVAL_AGES = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]


class DugongsEnv:
    def __init__(self):
        self.alpha = 2.65
        self.beta = 0.97
        self.lam = -0.87
        self.rng = random.Random()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng.seed(seed)

    def query(self, query: str) -> str:
        age = parse_query_float(query, "age")
        if age is not None:
            length = self.alpha - self.beta * abs(self.lam) ** age
            noise = self.rng.gauss(0, 0.1)
            return f"length={length + noise:.3f}"
        return "Invalid query. Use format: age=<number>"

    @staticmethod
    def get_lazyppl_columns() -> list[dict[str, str]]:
        return list(_LAZYPPL_COLUMNS)

    @staticmethod
    def get_lazyppl_prompt_info() -> dict[str, str]:
        return {
            "query_example": "age=3",
            "query_description": "Query with age=<number> to observe length",
            "row_fields_doc": "`xVal r` (Double, age) and `yVal r` (Double, observed length)",
        }

    @staticmethod
    def get_lazyppl_eval_data(seed: int = 7) -> list[dict]:
        """Generate 10 evaluation observations at fixed ages."""
        alpha, beta, lam = 2.65, 0.97, -0.87
        rng = random.Random(seed)
        rows = []
        for age in _EVAL_AGES:
            length = alpha - beta * abs(lam) ** age + rng.gauss(0, 0.1)
            rows.append({"xVal": float(age), "yVal": round(length, 4)})
        return rows
