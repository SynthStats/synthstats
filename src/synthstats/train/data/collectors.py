"""Trajectory collectors for training."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, cast, runtime_checkable

import torch

from synthstats.core.constants import DEFAULT_LANGUAGE, LOG_REWARD_CLAMP_RANGE
from synthstats.core.types import Action, FinalAnswer, Program, ToolCall
from synthstats.train.data.types import BufferEntry

PolicySample = tuple[dict[str, Any], float | torch.Tensor, float | torch.Tensor]
PolicySampleWithEOS = tuple[
    dict[str, Any], float | torch.Tensor, float | torch.Tensor, float | torch.Tensor | None
]
PolicyFnNoTemp = Callable[[str], PolicySample | PolicySampleWithEOS]
PolicyFnWithTemp = Callable[[str, float], PolicySample | PolicySampleWithEOS]
PolicyFn = PolicyFnNoTemp | PolicyFnWithTemp
ScoreResult = tuple[float | torch.Tensor, float | torch.Tensor]
ScoreResultWithEOS = tuple[float | torch.Tensor, float | torch.Tensor, float | torch.Tensor | None]
ScoreFnNoTemp = Callable[[str, dict[str, Any]], ScoreResult | ScoreResultWithEOS]
ScoreFnWithTemp = Callable[[str, dict[str, Any], float], ScoreResult | ScoreResultWithEOS]
ScoreFn = ScoreFnNoTemp | ScoreFnWithTemp


class EnvStepOutput(TypedDict):
    """Return type for CollectorEnv.step()."""

    observations: list[dict[str, str]]
    reward: float
    done: bool
    metadata: dict[str, Any]


@runtime_checkable
class CollectorEnv(Protocol):
    """Minimal environment interface required by TrajectoryCollector.

    ``init()`` resets the environment and returns the initial conversation.
    ``step(action_text)`` advances the environment by one action and returns
    an ``EnvStepOutput`` with ``reward``, ``done``, ``observations``, and ``metadata``.

    Optional attributes:
    - ``chat_history``: accumulated conversation (used for observation extraction).
    - ``codec``: action codec with a ``render`` method (used for action rendering).
    """

    def init(self) -> tuple[list[dict[str, str]], Any]: ...
    def step(self, action: str) -> EnvStepOutput: ...


@dataclass
class CollectedTrajectory:
    """Trajectory with log_probs, entropy, and optional ref/EOS."""

    observations: list[str]
    actions: list[dict[str, Any]]
    log_probs: torch.Tensor  # [T]
    entropy: torch.Tensor  # [T]
    reward: float
    ref_log_probs: torch.Tensor | None = None
    temperature: float = 1.0
    eos_logprobs: torch.Tensor | None = None  # [T] log P(EOS|state)
    prompts: list[str] | None = None
    completions: list[str] | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    def detach(self) -> CollectedTrajectory:
        ref_lp = self.ref_log_probs.detach().cpu() if self.ref_log_probs is not None else None
        eos_lp = self.eos_logprobs.detach().cpu() if self.eos_logprobs is not None else None
        return CollectedTrajectory(
            observations=self.observations,
            actions=self.actions,
            log_probs=self.log_probs.detach().cpu(),
            ref_log_probs=ref_lp,
            entropy=self.entropy.detach().cpu(),
            reward=self.reward,
            temperature=self.temperature,
            eos_logprobs=eos_lp,
            prompts=self.prompts,
            completions=self.completions,
            artifacts=dict(self.artifacts),
        )


def _accepts_parameter(fn: Callable[..., Any], param_name: str) -> bool:
    """Check if *fn* accepts a parameter named *param_name*."""
    try:
        sig = inspect.signature(fn)
        return param_name in sig.parameters
    except (ValueError, TypeError):
        return False


class TrajectoryCollector:
    """Runs episodes and collects trajectories with policy log-probs."""

    def __init__(
        self,
        env: CollectorEnv,
        policy_fn: PolicyFn,
        score_fn: ScoreFn | None = None,
    ) -> None:
        self.env = env
        self.policy_fn = policy_fn

        if score_fn is None and hasattr(policy_fn, "score_action"):
            self.score_fn = policy_fn.score_action
            self._score_fn_is_default = True
        else:
            self.score_fn = score_fn
            self._score_fn_is_default = False

        self._policy_accepts_temp = _accepts_parameter(policy_fn, "temperature")

        if self.score_fn is not None:
            self._score_accepts_temp = _accepts_parameter(self.score_fn, "temperature")
        else:
            self._score_accepts_temp = False

    def collect(
        self,
        episodes: int = 1,
        *,
        temperature: float = 1.0,
        compute_ref_log_probs: bool = False,
    ) -> list[CollectedTrajectory]:
        """Run `episodes` episodes, optionally scoring against a ref policy."""
        self._validate_collect_request(compute_ref_log_probs)
        trajectories = [
            self._collect_single_episode(
                temperature=temperature,
                compute_ref_log_probs=compute_ref_log_probs,
            )
            for _ in range(episodes)
        ]
        return trajectories

    def _validate_collect_request(self, compute_ref_log_probs: bool) -> None:
        if not compute_ref_log_probs:
            return
        if self.score_fn is None:
            raise ValueError("compute_ref_log_probs requires score_fn")
        if self._score_fn_is_default:
            raise ValueError("compute_ref_log_probs requires explicit ref policy score_fn")

    def _collect_single_episode(
        self,
        *,
        temperature: float,
        compute_ref_log_probs: bool,
    ) -> CollectedTrajectory:
        chat_history, _ = self.env.init()
        obs = self._extract_observation(chat_history)

        observations: list[str] = []
        actions: list[dict[str, Any]] = []
        log_probs: list[torch.Tensor] = []
        ref_log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        eos_log_probs: list[torch.Tensor] = []
        total_reward = 0.0

        done = False
        while not done:
            action, logp, ent, eos_logp = self._sample_action(obs, temperature)
            result = self._step_env(obs, action, temperature, compute_ref_log_probs)

            observations.append(obs)
            actions.append(action)
            log_probs.append(self._to_tensor(logp))
            entropies.append(self._to_tensor(ent))

            if eos_logp is not None:
                eos_log_probs.append(self._to_tensor(eos_logp))
            if result["ref_logp"] is not None:
                ref_log_probs.append(result["ref_logp"])

            total_reward += float(result["reward"])
            done = bool(result["done"])
            if not done:
                obs = self._resolve_next_observation(result["env_result"])

        eos_logprobs_t = self._finalize_eos_logprobs(eos_log_probs, log_probs)
        return CollectedTrajectory(
            observations=observations,
            actions=actions,
            log_probs=torch.stack(log_probs) if log_probs else torch.zeros(0),
            ref_log_probs=torch.stack(ref_log_probs) if compute_ref_log_probs else None,
            entropy=torch.stack(entropies) if entropies else torch.zeros(0),
            reward=total_reward,
            temperature=temperature,
            eos_logprobs=eos_logprobs_t,
        )

    def _step_env(
        self,
        obs: str,
        action: dict[str, Any],
        temperature: float,
        compute_ref_log_probs: bool,
    ) -> dict[str, Any]:
        ref_logp_t: torch.Tensor | None = None
        if compute_ref_log_probs:
            with torch.no_grad():
                ref_logp, _, _ = self._score_action(obs, action, temperature)
            ref_logp_t = self._to_tensor(ref_logp).detach()

        action_text = self._render_action(action)
        env_result = self.env.step(action_text)
        return {
            "env_result": env_result,
            "reward": env_result["reward"],
            "done": env_result["done"],
            "ref_logp": ref_logp_t,
        }

    def _resolve_next_observation(self, step_result: dict[str, Any]) -> str:
        chat_hist = getattr(self.env, "chat_history", None)
        if chat_hist:
            return self._extract_observation(chat_hist)
        if step_result["observations"]:
            return self._extract_observation(step_result["observations"])
        raise ValueError("Environment returned no observation for a non-terminal step")

    @staticmethod
    def _finalize_eos_logprobs(
        eos_logps: list[torch.Tensor],
        logps: list[torch.Tensor],
    ) -> torch.Tensor | None:
        if not eos_logps:
            return None
        if len(eos_logps) == len(logps):
            return torch.stack(eos_logps)

        import warnings

        warnings.warn(
            f"partial EOS logprobs ({len(eos_logps)}/{len(logps)}), discarding",
            UserWarning,
            stacklevel=2,
        )
        return None

    def replay_entry(
        self,
        entry: BufferEntry,
        temperature: float = 1.0,
    ) -> CollectedTrajectory | None:
        """Re-score a buffer entry with current policy for on-policy gradients."""
        if self.score_fn is None:
            raise ValueError("replay_entry requires score_fn")

        if len(entry.actions) == 0 or len(entry.observations) == 0:
            return None

        n_actions = len(entry.actions)
        observations = entry.observations[:n_actions]
        if len(observations) < n_actions:
            observations = observations + [observations[-1]] * (n_actions - len(observations))

        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        eos_logps: list[torch.Tensor] = []

        for obs, action in zip(observations, entry.actions, strict=True):
            logp, ent, eos_logp = self._score_action(obs, action, temperature)
            log_probs.append(self._to_tensor(logp))
            entropies.append(self._to_tensor(ent))

            if eos_logp is not None:
                eos_logps.append(self._to_tensor(eos_logp))

        # log→exp→log round-trip: BufferEntry stores log_reward from collection time,
        # but CollectedTrajectory.reward is a raw float that downstream consumers
        # re-log-transform. Minor floating-point drift is acceptable.
        clamped = max(min(entry.log_reward, LOG_REWARD_CLAMP_RANGE[1]), LOG_REWARD_CLAMP_RANGE[0])
        reward = math.exp(clamped)

        # only include eos_logprobs if we got one for every action
        eos_logprobs_t: torch.Tensor | None = None
        if eos_logps and len(eos_logps) == len(log_probs):
            eos_logprobs_t = torch.stack(eos_logps)

        return CollectedTrajectory(
            observations=observations,
            actions=list(entry.actions),
            log_probs=torch.stack(log_probs) if log_probs else torch.zeros(0),
            entropy=torch.stack(entropies) if entropies else torch.zeros(0),
            reward=reward,
            temperature=temperature,
            eos_logprobs=eos_logprobs_t,
            artifacts=dict(entry.artifacts),
        )

    @staticmethod
    def _extract_observation(messages: list[dict[str, str]]) -> str:
        import json

        return json.dumps(messages)

    def _render_action(self, action: dict[str, Any] | Action | str) -> str:
        if isinstance(action, str):
            return action
        if isinstance(action, Action):
            return self._render_with_codec(action)
        if isinstance(action, dict):
            parsed = self._dict_to_action(action)
            if parsed is not None:
                return self._render_with_codec(parsed)
        return str(action)

    def _render_with_codec(self, action: Action) -> str:
        codec = getattr(self.env, "codec", None)
        if codec is None:
            return str(action)
        if hasattr(codec, "render"):
            return codec.render(action)
        return str(action)

    @staticmethod
    def _dict_to_action(action: dict[str, Any]) -> Action | None:
        parsers = (
            TrajectoryCollector._parse_query_shorthand,
            TrajectoryCollector._parse_tool_call,
            TrajectoryCollector._parse_answer,
            TrajectoryCollector._parse_program,
            TrajectoryCollector._parse_typed_action,
        )
        for parser in parsers:
            parsed = parser(action)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _parse_query_shorthand(action: dict[str, Any]) -> Action | None:
        if "query" in action and len(action) <= 2 and "name" not in action:
            return ToolCall(name="query", input={"query": str(action["query"])}, raw="")
        return None

    @staticmethod
    def _parse_tool_call(action: dict[str, Any]) -> Action | None:
        if "tool" not in action and "name" not in action:
            return None

        name = action.get("tool") or action.get("name")
        if not name:
            return None
        raw_input = action.get("input", action.get("args", {}))
        if isinstance(raw_input, dict):
            input_payload = raw_input
        elif isinstance(raw_input, str):
            input_payload = {"query": raw_input} if name == "query" else {"value": raw_input}
        else:
            input_payload = {}
        return ToolCall(name=str(name), input=input_payload, raw="")

    @staticmethod
    def _parse_answer(action: dict[str, Any]) -> Action | None:
        if "answer" in action:
            return FinalAnswer(text=str(action["answer"]))
        return None

    @staticmethod
    def _parse_program(action: dict[str, Any]) -> Action | None:
        if "program" not in action and "code" not in action:
            return None
        code = action.get("program", action.get("code", ""))
        language = action.get("language", DEFAULT_LANGUAGE)
        return Program(code=str(code), language=str(language))

    @staticmethod
    def _parse_typed_action(action: dict[str, Any]) -> Action | None:
        if "type" not in action:
            return None

        action_type = str(action.get("type"))
        payload = action.get("payload", "")
        if action_type in {"answer", "final", "final_answer"}:
            return FinalAnswer(text=str(payload))
        if action_type in {"program", "submit_program"}:
            return Program(code=str(payload), language=action.get("language", DEFAULT_LANGUAGE))
        return ToolCall(
            name=action_type,
            input=TrajectoryCollector._typed_payload(action_type, payload),
            raw="",
        )

    @staticmethod
    def _typed_payload(action_type: str, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str) and action_type == "query":
            return {"query": payload}
        if isinstance(payload, str) and payload:
            return {"value": payload}
        return {}

    @staticmethod
    def _to_tensor(val: float | torch.Tensor) -> torch.Tensor:
        if isinstance(val, torch.Tensor):
            if val.dim() != 0:
                val = val.reshape(())
            return val
        return torch.tensor(float(val), dtype=torch.float32)

    def _sample_action(
        self,
        obs: str,
        temperature: float,
    ) -> tuple[
        dict[str, Any], float | torch.Tensor, float | torch.Tensor, float | torch.Tensor | None
    ]:
        # Two policy API shapes are supported:
        #   Shape A (preferred): policy_fn has .sample_with_eos(obs[, temp])
        #     → returns PolicySampleWithEOS = (action, logp, entropy, eos_logprob)
        #   Shape B (direct):    policy_fn(obs[, temp])
        #     → returns PolicySample = (action, logp, entropy) or PolicySampleWithEOS
        # Temperature is passed when the callable accepts ≥2 args (detected at init).
        sample_with_eos = getattr(self.policy_fn, "sample_with_eos", None)
        if callable(sample_with_eos):
            if _accepts_parameter(sample_with_eos, "temperature"):
                out = sample_with_eos(obs, temperature)
            else:
                out = sample_with_eos(obs)
            return self._unpack_policy_output(out)

        if self._policy_accepts_temp:
            out = cast(PolicyFnWithTemp, self.policy_fn)(obs, temperature)
        else:
            out = cast(PolicyFnNoTemp, self.policy_fn)(obs)
        return self._unpack_policy_output(out)

    def _score_action(
        self,
        obs: str,
        action: dict[str, Any],
        temperature: float,
    ) -> tuple[float | torch.Tensor, float | torch.Tensor, float | torch.Tensor | None]:
        if self.score_fn is None:
            raise ValueError("_score_action requires score_fn")

        score_with_eos = getattr(self.score_fn, "score_action_with_eos", None)
        if callable(score_with_eos):
            if _accepts_parameter(score_with_eos, "temperature"):
                out = score_with_eos(obs, action, temperature)
            else:
                out = score_with_eos(obs, action)
            return self._unpack_score_output(out)

        score_owner = getattr(self.score_fn, "__self__", None)
        owner_score_with_eos = getattr(score_owner, "score_action_with_eos", None)
        if callable(owner_score_with_eos):
            if _accepts_parameter(owner_score_with_eos, "temperature"):
                out = owner_score_with_eos(obs, action, temperature)
            else:
                out = owner_score_with_eos(obs, action)
            return self._unpack_score_output(out)

        if self._score_accepts_temp:
            out = cast(ScoreFnWithTemp, self.score_fn)(obs, action, temperature)
        else:
            out = cast(ScoreFnNoTemp, self.score_fn)(obs, action)
        return self._unpack_score_output(out)

    @staticmethod
    def _unpack_policy_output(
        out: PolicySample | PolicySampleWithEOS,
    ) -> tuple[
        dict[str, Any], float | torch.Tensor, float | torch.Tensor, float | torch.Tensor | None
    ]:
        if not isinstance(out, tuple):
            raise ValueError("policy_fn must return tuple(action, logp, entropy[, eos_logprob])")
        if len(out) == 3:
            action, logp, ent = out
            return cast(dict[str, Any], action), logp, ent, None
        if len(out) == 4:
            action, logp, ent, eos_logp = out
            return cast(dict[str, Any], action), logp, ent, eos_logp
        raise ValueError(
            f"policy_fn returned unsupported tuple length {len(out)}; expected 3 or 4 elements"
        )

    @staticmethod
    def _unpack_score_output(
        out: ScoreResult | ScoreResultWithEOS,
    ) -> tuple[float | torch.Tensor, float | torch.Tensor, float | torch.Tensor | None]:
        if not isinstance(out, tuple):
            raise ValueError("score_fn must return tuple(logp, entropy[, eos_logprob])")
        if len(out) == 2:
            logp, ent = out
            return logp, ent, None
        if len(out) == 3:
            logp, ent, eos_logp = out
            return logp, ent, eos_logp
        raise ValueError(
            f"score_fn returned unsupported tuple length {len(out)}; expected 2 or 3 elements"
        )
