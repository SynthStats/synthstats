"""Task for statistical modelling workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from synthstats.core.types import Action, Message, Program, StepResult, ToolCall

logger = logging.getLogger(__name__)

_LAZYPPL_SYSTEM_PROMPT_TEMPLATE = """\
You are a scientist discovering a statistical model for {env_name}.

## Module contract

Your submission MUST be a Haskell module with exactly this structure:

```haskell
module Submission where
import LazyPPL
import LazyPPL.Distributions
import TaskData (Row(..))

data State = State {{ ... }} deriving (Show)

model :: [Row] -> Meas State
model rows = do
  -- sample priors, then score against each row
  ...

logLikRow :: State -> Row -> Double
logLikRow state row = ...
```

Three exports: `State` (type), `model` (prior + scoring), `logLikRow` (pointwise log-likelihood).

### Import rules
- MUST import `LazyPPL`, `LazyPPL.Distributions`, and `TaskData (Row(..))`.
- Must NOT import `LazyPPLHarness`.
- Must NOT call inference functions (`mh`, `mhirreducible`, `lwis`) or any IO actions.
- Must NOT define `main`.

### Row fields
`Row` is imported with `(..)` so record fields are accessed as functions.
Available fields for this environment: {row_fields_doc}

### score convention
`score` takes a **positive Double** (the raw likelihood density), NOT a log-likelihood.
A score of 0 encodes impossibility.

### logLikRow convention
`logLikRow` returns a **log-density** (Double, typically negative).
Use the log-density functions listed below.

## Density function catalog

Sampling (in `model`, via `sample $`):
  normal mu sigma, gamma shape scale, beta a b, exponential rate,
  bernoulli p, categorical weights, poisson rate, uniformbounded lo hi,
  dirichlet alphas

Raw density (for `score`):
  normalPdf mu sigma x, expPdf rate x, poissonPdf rate k

Log-density (for `logLikRow`):
  normalLogPdf mu sigma x, exponentialLogPdf rate x,
  gammaLogPdf shape scale x, betaLogPdf a b x,
  bernoulliLogPmf p x, categoricalLogPmf probs k,
  uniformBoundedLogPdf lo hi x, poissonLogPmf rate k,
  dirichletLogPdf alphas xs, logNormalLogPdf mu sigma x,
  binomialLogPmf n p k

## Worked examples

### Regression (continuous y)
```haskell
module Submission where
import LazyPPL
import LazyPPL.Distributions
import TaskData (Row(..))

data State = State {{ slopeA :: Double, interceptB :: Double, noiseS :: Double }}
  deriving (Show)

model :: [Row] -> Meas State
model rows = do
  a <- sample $ normal 0 3
  b <- sample $ normal 0 3
  s <- sample $ gamma 2 1
  mapM_ (\\r -> score $ normalPdf (a * xVal r + b) s (yVal r)) rows
  return $ State a b s

logLikRow :: State -> Row -> Double
logLikRow (State a b s) r =
  normalLogPdf (a * xVal r + b) s (yVal r)
```

### Poisson rate (integer counts)
```haskell
module Submission where
import LazyPPL
import LazyPPL.Distributions
import TaskData (Row(..))

data State = State {{ lambda :: Double }} deriving (Show)

model :: [Row] -> Meas State
model rows = do
  lam <- sample $ gamma 2 2
  mapM_ (\\r -> score $ poissonPdf lam (count r)) rows
  return $ State lam

logLikRow :: State -> Row -> Double
logLikRow (State lam) r =
  poissonLogPmf lam (count r)
```

## Key pattern
Use `mapM_` with `score` to loop over rows in `model`.
Use the corresponding log-density function in `logLikRow`.\
"""


@dataclass
class StatisticalModelingState:
    observations: list[str] = field(default_factory=list)
    step: int = 0
    done: bool = False


class StatisticalModelingTask:
    """Task plugin for general statistical modelling workflows."""

    name = "statistical_modeling"
    state_type = StatisticalModelingState

    def __init__(
        self,
        env_name: str = "dugongs",
        max_steps: int = 10,
        language: str = "pymc",
    ) -> None:
        self.env_name = env_name
        self.max_steps = max_steps
        self.language = language
        self.env = self._load_env(env_name)

    def _load_env(self, env_name: str):
        if env_name == "dugongs":
            from synthstats.tasks.statistical_modeling.envs.dugongs_env import DugongsEnv

            return DugongsEnv()
        if env_name == "peregrines":
            from synthstats.tasks.statistical_modeling.envs.peregrines_env import PeregrinesEnv

            return PeregrinesEnv()
        if env_name == "eight_schools":
            from synthstats.tasks.statistical_modeling.envs.eight_schools_env import EightSchoolsEnv

            return EightSchoolsEnv()
        if env_name == "surgical":
            from synthstats.tasks.statistical_modeling.envs.surgical_env import SurgicalEnv

            return SurgicalEnv()
        raise ValueError(
            f"Unknown environment: {env_name}. "
            "Supported environments: dugongs, eight_schools, peregrines, surgical."
        )

    def _make_state(
        self,
        *,
        observations: list[str],
        step: int,
        done: bool,
    ) -> StatisticalModelingState:
        return self.state_type(observations=observations, step=step, done=done)

    def native_actions(self) -> frozenset[str]:
        return frozenset({"query"})

    def reset(self, seed: int | None = None) -> StatisticalModelingState:
        self.env.reset(seed=seed)
        return self._make_state(observations=[], step=0, done=False)

    def tool_schemas(self) -> list[dict]:
        from synthstats.tasks.statistical_modeling.tools import get_tool_schemas

        return get_tool_schemas(self.env_name)

    def system_prompt(self) -> str:
        if self.language == "lazyppl":
            return self._lazyppl_system_prompt()
        return self._pymc_system_prompt()

    def _pymc_system_prompt(self) -> str:
        return (
            f"You are a scientist discovering a statistical model for {self.env_name}.\n"
            "\n"
            "Use the `query` tool to collect data from the environment, "
            "then use the `submit` tool to submit a PyMC program.\n"
            "\n"
            "Your program should define a PyMC model using the observed data."
        )

    def _lazyppl_system_prompt(self) -> str:
        prompt_info = getattr(self.env, "get_lazyppl_prompt_info", lambda: None)()
        if prompt_info is None:
            prompt_info = {
                "query_example": "age=3",
                "query_description": "Query the environment",
                "row_fields_doc": "`xVal r`, `yVal r`, etc.",
            }
        return _LAZYPPL_SYSTEM_PROMPT_TEMPLATE.format(
            env_name=self.env_name,
            **prompt_info,
        )

    def observe(self, state: StatisticalModelingState) -> list[Message]:
        system_msg = Message(role="system", content=self.system_prompt())
        obs_content = (
            "\n".join(state.observations) if state.observations else "No observations yet."
        )
        user_msg = Message(role="user", content=f"Observations:\n{obs_content}")
        return [system_msg, user_msg]

    def step(self, state: StatisticalModelingState, action: Action) -> StepResult:
        if isinstance(action, Program):
            next_state = self._make_state(
                observations=list(state.observations),
                step=state.step + 1,
                done=True,
            )
            return StepResult(
                next_state=next_state,
                done=True,
                artifacts={"program": action.code, "language": action.language},
            )
        if isinstance(action, ToolCall) and action.name == "query":
            query = action.input.get("query", "")
            result = self.env.query(query)
            new_observations = list(state.observations)
            new_observations.append(f"Query: {query}\nResult: {result}")
            new_step = state.step + 1
            done = new_step >= self.max_steps
            next_state = self._make_state(
                observations=new_observations,
                step=new_step,
                done=done,
            )
            return StepResult(
                next_state=next_state,
                done=done,
                artifacts={},
            )
        return StepResult(next_state=state, done=False, artifacts={})


__all__ = ["StatisticalModelingState", "StatisticalModelingTask"]
