#!/usr/bin/env python3
"""Generate LazyPPL programs for pilot environments and convert to ATIF trajectories.

For each pilot environment (dugongs, peregrines, eight_schools, surgical), prompts
a local Qwen3.5-0.8B model to generate Haskell programs implementing the required
(State, model, logLikRow) interface. Compiling programs are saved as ATIF v1.6 JSON
trajectory files.

Usage:
    uv run python scripts/generate_pilot_programs.py
    uv run python scripts/generate_pilot_programs.py --env dugongs --n 2
    uv run python scripts/generate_pilot_programs.py --env dugongs --n 5 --no-typecheck
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from _utils import (
    ATIF_SCHEMA_VERSION,
    build_atif_agent,
    pilot_data_dir,
    typecheck,
    write_json_atomic,
)
from sft_to_atif import validate_atif_record

from synthstats.core.reward_semantics import annotate_reward_info

log = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"

ALL_ENVS = ["dugongs", "peregrines", "eight_schools", "surgical"]


class GenerationRecord(TypedDict):
    text: str
    token_ids: list[int]
    logprobs: list[float]


@dataclass(frozen=True)
class GenerationSettings:
    n: int
    temperature: float
    top_p: float
    max_new_tokens: int


@dataclass(frozen=True)
class RunConfig:
    envs: list[str]
    model_id: str
    use_api: bool
    no_typecheck: bool
    generation: GenerationSettings
    api_base: str | None
    api_key: str | None
    pilot_data_dir: Path
    output_dir: Path


@dataclass
class RunStats:
    total_generated: int = 0
    total_extracted: int = 0
    total_compiled: int = 0


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_output_dir() -> Path:
    return project_root() / "data" / "atif_trajectories"


def default_pilot_data_dir() -> Path:
    return pilot_data_dir()


def load_pilot_task(env: str, *, pilot_data_dir: Path) -> dict[str, Any]:
    path = pilot_data_dir / f"lazyppl_pilot_{env}.json"
    if not path.exists():
        raise FileNotFoundError(f"Pilot data not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_envs(selected_env: str | None) -> list[str]:
    if selected_env is None:
        return ALL_ENVS
    if selected_env not in ALL_ENVS:
        raise ValueError(f"Unknown environment: {selected_env} (valid: {ALL_ENVS})")
    return [selected_env]


def build_run_config(
    parsed_args: argparse.Namespace,
    *,
    pilot_data_dir: Path | None = None,
    output_dir: Path | None = None,
) -> RunConfig:
    return RunConfig(
        envs=resolve_envs(parsed_args.env),
        model_id=parsed_args.model,
        use_api=parsed_args.api,
        no_typecheck=parsed_args.no_typecheck,
        generation=GenerationSettings(
            n=parsed_args.n,
            temperature=parsed_args.temperature,
            top_p=parsed_args.top_p,
            max_new_tokens=parsed_args.max_new_tokens,
        ),
        api_base=parsed_args.api_base,
        api_key=parsed_args.api_key,
        pilot_data_dir=pilot_data_dir or default_pilot_data_dir(),
        output_dir=output_dir
        or (Path(parsed_args.output_dir) if parsed_args.output_dir else default_output_dir()),
    )


def load_pilot_tasks(envs: list[str], *, pilot_data_dir: Path) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for env in envs:
        tasks[env] = load_pilot_task(env, pilot_data_dir=pilot_data_dir)
        log.info("Loaded pilot task: %s (%d obs)", env, tasks[env]["n_observations"])
    return tasks


# ── Prompt construction ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert Haskell probabilistic programmer using the LazyPPL library.
Given a statistical modelling task with observed data, write a LazyPPL module that exports:
  1. A `State` data type holding posterior parameters.
  2. A `model :: [Row] -> Meas State` function that defines priors (via `sample`)
     and scores each observation row (via `score`).
  3. A `logLikRow :: State -> Row -> Double` function returning the per-observation
     log-likelihood given posterior parameters.

The Row type is provided by a separate TaskData module (you do NOT define it).
Import TaskData (Row(..)) to access the row fields.

Your module header must be: module Submission where

Available imports:
  import LazyPPL
  import LazyPPL.Distributions
  import TaskData (Row(..))

Key LazyPPL API:
  sample :: Prob a -> Meas a          -- draw from a prior
  score  :: Double -> Meas ()         -- multiply by a likelihood weight
  normal :: Double -> Double -> Prob Double
  gamma  :: Double -> Double -> Prob Double
  beta   :: Double -> Double -> Prob Double
  exponential :: Double -> Prob Double
  uniform :: Prob Double              -- Uniform(0,1)
  normalLogPdf :: Double -> Double -> Double -> Double
  gammaLogPdf  :: Double -> Double -> Double -> Double
  betaLogPdf   :: Double -> Double -> Double -> Double
  poissonLogPmf :: Double -> Integer -> Double
  exponentialLogPdf :: Double -> Double -> Double
  bernoulliLogPmf :: Double -> Bool -> Double
  binomialLogPmf :: Int -> Double -> Int -> Double"""


EXAMPLE_PROGRAM = """\
-- Example: linear regression on (xVal :: Double, yVal :: Double) data
module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData (Row(..))

data State = State
  { slope     :: Double
  , intercept :: Double
  , sigma     :: Double
  } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  m <- sample (normal 0 10)
  b <- sample (normal 0 10)
  s <- sample (gamma 2 1)
  let st = State m b s
  mapM_ (\\r -> score (exp (logLikRow st r))) rows
  return st

logLikRow :: State -> Row -> Double
logLikRow st r =
  let mu = slope st * xVal r + intercept st
  in  normalLogPdf mu (sigma st) (yVal r)"""


def build_user_prompt(task: dict) -> str:
    """Build the user prompt from a pilot task JSON."""
    desc = task["description"]
    cols = task["columns"]
    obs = task["observations"]
    n_obs = task["n_observations"]

    col_schema = ", ".join(f"{c['name']} :: {c['hs_type']}" for c in cols)

    sample_rows = obs[: min(3, len(obs))]
    sample_str = "\n".join(
        "  " + ", ".join(f"{c['name']}={row[c['name']]}" for c in cols) for row in sample_rows
    )

    return f"""\
Task: {desc}

Data schema (Row record fields): {col_schema}
Number of observations: {n_obs}

Sample data:
{sample_str}

Here is a worked example for a simple linear regression task:

```haskell
{EXAMPLE_PROGRAM}
```

Now write a Submission module for the task above. The module must export
State, model, and logLikRow. Wrap your code in ```haskell ... ``` markers."""


# ── Code extraction ─────────────────────────────────────────────────


def extract_haskell_code(text: str) -> str | None:
    """Pull code from ```haskell...``` fences, or fall back to module-based extraction."""
    # try fenced code block first
    m = re.search(r"```haskell\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # fall back: find from module/import to end
    m = re.search(r"(module\s+\S+\s+where\b.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    return None


# ── ATIF construction ───────────────────────────────────────────────


def build_atif(
    env_name: str,
    idx: int,
    system_msg: str,
    user_msg: str,
    completion: str,
    code: str,
    token_ids: list[int],
    logprobs: list[float],
    compiles: bool,
    reward_source: str,
    model_name: str = DEFAULT_MODEL_ID,
) -> dict[str, Any]:
    """Build an ATIF v1.6 trajectory dict."""
    session_id = f"pilot-{env_name}-{idx:04d}"

    agent_step: dict = {
        "step_id": 3,
        "source": "agent",
        "message": completion,
        "tool_calls": [
            {
                "tool_call_id": "tc_001",
                "function_name": "submit_program",
                "arguments": {"language": "haskell", "code": code},
            }
        ],
        "metrics": {
            "completion_tokens": len(token_ids),
            "completion_token_ids": token_ids,
            "logprobs": logprobs,
        },
    }

    final_metrics = annotate_reward_info(
        {
            "reward": 1.0 if compiles else 1e-8,
            "compiles": compiles,
            "reward_source": reward_source,
        },
        reward_source=reward_source,
    )

    atif = {
        "schema_version": ATIF_SCHEMA_VERSION,
        "session_id": session_id,
        "agent": build_atif_agent(model_name),
        "steps": [
            {"step_id": 1, "source": "system", "message": system_msg},
            {"step_id": 2, "source": "user", "message": user_msg},
            agent_step,
        ],
        "final_metrics": {"extra": final_metrics},
    }
    validate_atif_record(atif)
    return atif


# ── Model loading and generation ────────────────────────────────────


def load_model_and_tokenizer(model_id: str) -> tuple[Any, Any]:
    """Load a HuggingFace model for local generation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading %s tokenizer...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    log.info("Loading %s model with device_map='auto'...", model_id)
    t0 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    log.info("Model loaded in %.1fs", time.monotonic() - t0)

    return model, tokenizer


def generate_programs(
    model: Any,
    tokenizer: Any,
    system_msg: str,
    user_msg: str,
    generation: GenerationSettings,
) -> list[GenerationRecord]:
    """Generate programs, returning {text, token_ids, logprobs} records."""
    import torch

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )
    # handle both dict-like and tensor returns
    if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
        input_ids = encoded["input_ids"].to(model.device)
    elif isinstance(encoded, torch.Tensor):
        input_ids = encoded.to(model.device)
    else:
        # list of token ids
        input_ids = torch.tensor([encoded], dtype=torch.long, device=model.device)
    input_len = input_ids.shape[1]

    results: list[GenerationRecord] = []
    for i in range(generation.n):
        log.info("  generating program %d/%d...", i + 1, generation.n)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=generation.max_new_tokens,
                temperature=generation.temperature,
                top_p=generation.top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        gen_ids = outputs.sequences[0, input_len:]
        gen_token_ids = gen_ids.tolist()

        # compute per-token logprobs from scores
        logprobs = []
        for step_idx, scores in enumerate(outputs.scores):
            probs = torch.softmax(scores[0], dim=-1)
            token_id = gen_token_ids[step_idx]
            lp = torch.log(probs[token_id] + 1e-30).item()
            logprobs.append(round(lp, 6))

        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        results.append(
            {
                "text": text,
                "token_ids": gen_token_ids,
                "logprobs": logprobs,
            }
        )

    return results


# ── API generation (OpenRouter / OpenAI-compatible) ────────────────


def generate_programs_api(
    model_id: str,
    system_msg: str,
    user_msg: str,
    generation: GenerationSettings,
    api_base: str | None = None,
    api_key: str | None = None,
) -> list[GenerationRecord]:
    """Generate programs via an OpenAI-compatible API (e.g., OpenRouter).

    Returns {text, token_ids, logprobs} records. token_ids and logprobs
    come from the API's logprobs response field.
    """
    import os

    base = api_base or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("No API key: set OPENROUTER_API_KEY or pass --api-key")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai package required for API generation. "
            "Install with: uv pip install 'synthstats[api]'"
        ) from exc

    client = OpenAI(base_url=base, api_key=key)
    results: list[GenerationRecord] = []

    for i in range(generation.n):
        log.info(
            "  generating program %d/%d via API (%s)...",
            i + 1,
            generation.n,
            model_id,
        )
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=generation.temperature,
            top_p=generation.top_p,
            max_tokens=generation.max_new_tokens,
            logprobs=True,
            top_logprobs=1,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""

        if not choice.logprobs or not choice.logprobs.content:
            raise ValueError("API response must include native token IDs and logprobs")
        token_ids: list[int] = []
        logprobs_list: list[float] = []
        for token in choice.logprobs.content:
            token_id = getattr(token, "token_id", None)
            if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                raise ValueError("API response has no model-native token IDs; use local generation")
            logprob = float(token.logprob)
            if not math.isfinite(logprob):
                raise ValueError("API response contains a non-finite logprob")
            token_ids.append(token_id)
            logprobs_list.append(logprob)

        results.append(
            {
                "text": text,
                "token_ids": token_ids,
                "logprobs": logprobs_list,
            }
        )

    return results


def build_generation_runner(
    config: RunConfig,
) -> Callable[[str], list[GenerationRecord]]:
    if config.use_api:

        def run_generation_api(user_msg: str) -> list[GenerationRecord]:
            return generate_programs_api(
                config.model_id,
                SYSTEM_PROMPT,
                user_msg,
                generation=config.generation,
                api_base=config.api_base,
                api_key=config.api_key,
            )

        return run_generation_api

    model, tokenizer = load_model_and_tokenizer(config.model_id)

    def run_generation_local(user_msg: str) -> list[GenerationRecord]:
        return generate_programs(
            model,
            tokenizer,
            SYSTEM_PROMPT,
            user_msg,
            generation=config.generation,
        )

    return run_generation_local


def typecheck_generation(
    env: str,
    idx: int,
    code: str,
    *,
    skip_typecheck: bool,
) -> bool:
    if skip_typecheck:
        return True

    module_name = f"PilotCheck{env.title().replace('_', '')}{idx:02d}"
    ok, output = typecheck(code, module_name)
    status = "OK" if ok else "FAIL"
    log.info("  [%s-%d] typecheck: %s", env, idx, status)
    if ok:
        return True

    for line in output.splitlines():
        if "error" in line.lower() and module_name in line:
            log.info("    %s", line.strip()[:120])
            break
    return False


def compilation_reward_source(*, skip_typecheck: bool) -> str:
    if skip_typecheck:
        return "heuristic.compilation_assumed"
    return "validation.compilation"


def process_generation(
    env: str,
    idx: int,
    generation: GenerationRecord,
    *,
    config: RunConfig,
    user_msg: str,
    stats: RunStats,
) -> None:
    stats.total_generated += 1
    code = extract_haskell_code(generation["text"])

    if code is None:
        log.warning("  [%s-%d] no Haskell code extracted", env, idx)
        return

    stats.total_extracted += 1
    compiles = typecheck_generation(
        env,
        idx,
        code,
        skip_typecheck=config.no_typecheck,
    )
    reward_source = compilation_reward_source(skip_typecheck=config.no_typecheck)
    if compiles:
        stats.total_compiled += 1

    atif = build_atif(
        env_name=env,
        idx=idx,
        system_msg=SYSTEM_PROMPT,
        user_msg=user_msg,
        completion=generation["text"],
        code=code,
        token_ids=generation["token_ids"],
        logprobs=generation["logprobs"],
        compiles=compiles,
        reward_source=reward_source,
        model_name=config.model_id,
    )

    out_path = config.output_dir / f"{atif['session_id']}.json"
    write_json_atomic(out_path, atif)
    log.info(
        "  [%s-%d] saved %s (reward=%.1e)",
        env,
        idx,
        out_path.name,
        atif["final_metrics"]["extra"]["reward"],
    )


# ── Main ────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate pilot LazyPPL programs as ATIF trajectories")
    p.add_argument(
        "--env",
        type=str,
        default=None,
        help="Single environment to process (default: all)",
    )
    p.add_argument("--n", type=int, default=5, help="Programs per environment (default: 5)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})",
    )
    p.add_argument("--no-typecheck", action="store_true", help="Skip GHC type-checking")
    p.add_argument(
        "--api",
        action="store_true",
        help="Use OpenRouter API instead of local HuggingFace model",
    )
    p.add_argument("--api-base", type=str, default=None, help="API base URL (default: OpenRouter)")
    p.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (default: OPENROUTER_API_KEY env var)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/atif_trajectories)",
    )
    return p.parse_args()


def main(
    *,
    args: argparse.Namespace | None = None,
    pilot_data_dir: Path | None = None,
    output_dir: Path | None = None,
) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parsed_args = args or parse_args()

    try:
        config = build_run_config(
            parsed_args,
            pilot_data_dir=pilot_data_dir,
            output_dir=output_dir,
        )
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    try:
        tasks = load_pilot_tasks(config.envs, pilot_data_dir=config.pilot_data_dir)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1

    try:
        generate_for_prompt = build_generation_runner(config)
    except Exception as exc:
        log.error("Failed to load model: %s", exc)
        log.error("Install transformers + torch, or use --api for API inference")
        return 1

    config.output_dir.mkdir(parents=True, exist_ok=True)
    stats = RunStats()

    for env, task in tasks.items():
        log.info("=== Environment: %s ===", env)
        user_msg = build_user_prompt(task)

        try:
            generations = generate_for_prompt(user_msg)
        except Exception as exc:
            log.error("Failed to generate programs for %s: %s", env, exc)
            return 1

        for i, generation in enumerate(generations):
            process_generation(
                env,
                i,
                generation,
                config=config,
                user_msg=user_msg,
                stats=stats,
            )

    log.info("")
    log.info(
        "Summary: %d generated, %d code extracted, %d compiled",
        stats.total_generated,
        stats.total_extracted,
        stats.total_compiled,
    )
    log.info("Output directory: %s", config.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
