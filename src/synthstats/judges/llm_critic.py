"""LLM-as-critic judge for process reward modelling."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from synthstats.core.reward_semantics import annotate_reward_info
from synthstats.core.types import Reward, Trajectory

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TEMPLATE = (
    "You are an expert statistician and probabilistic programmer reviewing PyMC code.\n"
    "\n"
    "TASK: {task_name}\n"
    "\n"
    "PROGRAM TO REVIEW:\n"
    "```python\n"
    "{program}\n"
    "```\n"
    "\n"
    "Rate this program on three dimensions (0.0 to 1.0):\n"
    "\n"
    "1. **code_quality**: Is the code well-structured, readable, and follows PyMC best practices?\n"
    "   - 1.0: Excellent structure, clear naming, proper indentation\n"
    "   - 0.5: Acceptable but could be improved\n"
    "   - 0.0: Poorly structured, unreadable, or has syntax errors\n"
    "\n"
    "2. **statistical_validity**: Is the statistical model sound and appropriate?\n"
    "   - 1.0: Well-specified priors, appropriate likelihood, valid model structure\n"
    "   - 0.5: Minor issues with priors or model specification\n"
    "   - 0.0: Invalid model, inappropriate distributions, or logical errors\n"
    "\n"
    "3. **problem_alignment**: Does the model address the stated problem effectively?\n"
    "   - 1.0: Directly models the phenomenon, captures key relationships\n"
    "   - 0.5: Partially addresses the problem\n"
    "   - 0.0: Does not address the problem or misses key aspects\n"
    "\n"
    "Respond ONLY in this exact format:\n"
    "SCORES:\n"
    "code_quality: <score>\n"
    "statistical_validity: <score>\n"
    "problem_alignment: <score>\n"
    "RATIONALE: <brief one-line explanation>"
)

LAZYPPL_PROMPT_TEMPLATE = (
    "You are an expert in Bayesian statistics and LazyPPL (Haskell) reviewing a "
    "probabilistic program.\n"
    "\n"
    "LazyPPL uses two monads: Prob (normalized distributions, sampling only) "
    "and Meas (unnormalized "
    "measures, supports score for conditioning). Key functions: sample :: Prob a -> Meas a lifts "
    "distributions into measures; score :: Double -> Meas () weights traces by likelihood. "
    "Inference runs via mh (Metropolis-Hastings), lwis, or weightedsamples.\n"
    "\n"
    "TASK: {task_name}\n"
    "\n"
    "PROGRAM TO REVIEW:\n"
    "```haskell\n"
    "{program}\n"
    "```\n"
    "\n"
    "Rate this program on three dimensions (0.0 to 1.0):\n"
    "\n"
    "1. **code_quality**: Idiomatic Haskell - proper use of do-notation, type signatures, "
    "clear naming, correct monad usage (Prob vs Meas).\n"
    "   - 1.0: Clean, well-typed, monadic code with appropriate type signatures\n"
    "   - 0.5: Works but awkward style or missing signatures\n"
    "   - 0.0: Wrong monad boundaries (e.g. score inside Prob), syntax errors, or unidiomatic\n"
    "\n"
    "2. **statistical_validity**: Correct Bayesian model - appropriate priors, "
    "proper use of sample/score, valid likelihood via PDF helpers (normalPdf, etc.).\n"
    "   - 1.0: Sound priors, correct conditioning with score, valid model structure\n"
    "   - 0.5: Minor issues (e.g. vague priors, missing observations)\n"
    "   - 0.0: Wrong distributions, score used incorrectly, or logically broken model\n"
    "\n"
    "3. **problem_alignment**: Does the model address the stated problem?\n"
    "   - 1.0: Directly models the phenomenon, captures key relationships\n"
    "   - 0.5: Partially addresses the problem\n"
    "   - 0.0: Does not address the problem or misses key aspects\n"
    "\n"
    "Respond ONLY in this exact format:\n"
    "SCORES:\n"
    "code_quality: <score>\n"
    "statistical_validity: <score>\n"
    "problem_alignment: <score>\n"
    "RATIONALE: <brief one-line explanation>"
)


@dataclass
class LLMCriticConfig:
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.3
    num_samples: int = 1
    prompt_template: str | None = None
    default_score: float = 0.5
    api_base: str | None = None
    api_key_env_var: str | None = None
    max_tokens: int | None = None
    timeout_s: float | None = None
    max_retries: int | None = None
    language: str = "pymc"


class LLMCriticUnavailableError(Exception):
    """The critic cannot score; this also aborts composite reward evaluation."""


class LLMCriticJudge:
    """LiteLLM-based judge scoring code quality, statistical validity and how well
    the program addresses the stated modelling problem."""

    _LANGUAGE_TEMPLATES: dict[str, str] = {
        "pymc": DEFAULT_PROMPT_TEMPLATE,
        "lazyppl": LAZYPPL_PROMPT_TEMPLATE,
    }

    def __init__(self, config: LLMCriticConfig | None = None):
        cfg = config or LLMCriticConfig()
        self.model_name = cfg.model_name
        self.temperature = cfg.temperature
        self.num_samples = max(1, cfg.num_samples)
        self.language = cfg.language
        if cfg.prompt_template is not None:
            self.prompt_template = cfg.prompt_template
        else:
            self.prompt_template = self._LANGUAGE_TEMPLATES.get(
                cfg.language,
                DEFAULT_PROMPT_TEMPLATE,
            )
        self.default_score = cfg.default_score
        self.api_base = cfg.api_base
        self.api_key_env_var = cfg.api_key_env_var
        self.max_tokens = max(1, int(cfg.max_tokens)) if cfg.max_tokens is not None else None
        self.timeout_s = cfg.timeout_s
        self.max_retries = cfg.max_retries

    def _build_prompt(self, program: str, task_name: str) -> str:
        return self.prompt_template.format(task_name=task_name, program=program)

    def _parse_critique(self, response: str) -> dict[str, float]:
        result = {
            "code_quality": self.default_score,
            "statistical_validity": self.default_score,
            "problem_alignment": self.default_score,
        }

        scores_match = re.search(
            r"SCORES:\s*(.*?)(?:RATIONALE:|$)", response, re.DOTALL | re.IGNORECASE
        )
        if not scores_match:
            return result

        scores_text = scores_match.group(1)

        for key in result:
            pattern = rf"{key}:\s*(-?[\d.]+)"
            match = re.search(pattern, scores_text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    result[key] = max(0.0, min(1.0, score))  # clamp to [0, 1]
                except ValueError:
                    logger.debug("Could not parse score for '%s' from response", key)

        return result

    def _call_llm(self, prompt: str) -> str | None:
        api_key = None
        if self.api_key_env_var:
            api_key = os.getenv(self.api_key_env_var)
            if not api_key:
                raise LLMCriticUnavailableError(
                    f"LLM critic requires the credential variable {self.api_key_env_var}"
                )
        try:
            import litellm
        except ImportError as exc:
            raise LLMCriticUnavailableError(
                "LLM critic requires LiteLLM in its execution environment"
            ) from exc

        try:
            kwargs: dict[str, Any] = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            if self.api_base:
                kwargs["api_base"] = self.api_base
            if api_key:
                kwargs["api_key"] = api_key
            if self.timeout_s is not None:
                kwargs["timeout"] = float(self.timeout_s)
            if self.max_retries is not None:
                kwargs["num_retries"] = int(self.max_retries)

            response = litellm.completion(**kwargs)
            return response.choices[0].message.content
        except Exception as exc:  # noqa: BLE001 - litellm raises varied types (APIError, Timeout, etc.)
            raise LLMCriticUnavailableError(
                f"LLM critic request failed ({type(exc).__name__})"
            ) from None

    def score(self, *, task_name: str, trajectory: Trajectory, artifacts: dict[str, Any]) -> Reward:
        program = artifacts.get("program", "")

        if not program.strip():
            return Reward(
                total=0.0,
                components={
                    "code_quality": 0.0,
                    "statistical_validity": 0.0,
                    "problem_alignment": 0.0,
                },
                info=annotate_reward_info(
                    {"error": "empty_program"},
                    source="llm_critic",
                ),
            )

        prompt = self._build_prompt(program, task_name)

        all_scores: list[dict[str, float]] = []
        responses: list[str] = []

        for _ in range(self.num_samples):
            response = self._call_llm(prompt)
            if not response or not response.strip():
                raise LLMCriticUnavailableError("LLM critic returned no critique")
            responses.append(response)
            scores = self._parse_critique(response)
            all_scores.append(scores)

        avg_scores = {
            "code_quality": sum(s["code_quality"] for s in all_scores) / len(all_scores),
            "statistical_validity": sum(s["statistical_validity"] for s in all_scores)
            / len(all_scores),
            "problem_alignment": sum(s["problem_alignment"] for s in all_scores) / len(all_scores),
        }

        total = sum(avg_scores.values()) / len(avg_scores)

        info: dict[str, Any] = {"num_samples": len(all_scores)}
        if len(responses) == 1:
            info["response"] = responses[0]

        return Reward(
            total=total,
            components=avg_scores,
            info=annotate_reward_info(info, source="llm_critic"),
        )
