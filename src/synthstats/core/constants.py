"""Shared constants for SynthStats."""

from __future__ import annotations

import math

# environment variables safe to pass to sandboxed subprocesses
# allowlist to prevent leaking API keys, credentials, etc.
# NOTE: PYTHONPATH intentionally excluded - could allow module injection
# from shared directories on multi-tenant systems
SAFE_ENV_VARS: frozenset[str] = frozenset(
    {
        "PATH",
        "HOME",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
    }
)

# reward floor: minimum reward before log-transform to avoid log(0)
REWARD_FLOOR_DEFAULT: float = 1e-4

# TB loss: max absolute residual (clamp for numerical stability)
TB_MAX_RESIDUAL_DEFAULT: float = 100.0

# SubTB: lambda decay for sub-trajectory lengths
SUBTB_LAMBDA_DEFAULT: float = 0.9

# logZ learning rate (canonical: 10x base LR of 0.001)
# matches gfn-lm-tuning paper and src/synthstats/configs/objective/tb_subtb.yaml
LOGZ_LR_DEFAULT: float = 0.01

# sparse reward for incomplete prefixes in endpoint SubTB
LOG_SPARSE_REWARD_DEFAULT: float = math.log(1e-4)

# fallback log-reward when ELPD is unavailable (e.g. SFT warmstart without compute_rewards)
LOG_REWARD_DEFAULT: float = -5.0

# default timeout for tool execution (query, compute_eig, etc.)
TOOL_TIMEOUT_S_DEFAULT: float = 30.0

# timeout for PyMC ELPD subprocess execution
PYMC_EXECUTOR_TIMEOUT_S: float = 180.0

# timeout for compilation checks (GHC type-check, AST validation)
COMPILATION_TIMEOUT_S: float = 60.0

# timeout for external API calls (Harbor, LLM judges, etc.)
API_TIMEOUT_S: float = 120.0

# default random seed
DEFAULT_SEED: int = 42

# high-priority training/policy defaults
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_LANGUAGE: str = "pymc"
DEFAULT_MAX_TURNS: int = 20
DEFAULT_LEARNING_RATE: float = 1e-5
DEFAULT_BATCH_SIZE: int = 4
DEFAULT_ENTROPY_COEF: float = 0.01
DEFAULT_MAX_NEW_TOKENS: int | None = None
LOG_REWARD_CLAMP_RANGE: tuple[float, float] = (-700.0, 700.0)


def validate_reward_floor(reward_floor: float) -> None:
    """Validate reward_floor is positive (shared across all config sites)."""
    if reward_floor <= 0:
        raise ValueError(f"reward_floor must be > 0, got {reward_floor}")


# common stop tokens by model family
STOP_TOKEN_IDS: dict[str, list[int]] = {
    "qwen3": [151643, 151645],  # <|endoftext|>, <|im_end|>
    "qwen2": [151643, 151645],
    "glm": [151329, 151336, 151338],  # <|endoftext|>, <|user|>, <|observation|>
    "llama": [2, 128001, 128009],  # Llama 2 </s>, Llama 3 <|end_of_text|>, <|eot_id|>
    "mistral": [2],
    "default": [2],
}
