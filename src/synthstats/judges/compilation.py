"""Compilation-based judge for LazyPPL programs."""

from __future__ import annotations

import logging
from typing import Any

from synthstats.core.constants import COMPILATION_TIMEOUT_S
from synthstats.core.reward_semantics import annotate_reward_info
from synthstats.core.runtime_registry import (
    default_runtime_name,
    get_runtime_spec,
    normalize_runtime_name,
)
from synthstats.core.types import Reward, Trajectory

logger = logging.getLogger(__name__)
DEFAULT_RUNTIME_NAME = default_runtime_name()


class CompilationJudge:
    """Returns 1.0 if the program compiles, 0.0 otherwise.

    Wraps LazyPPLExecutor._compile() for Haskell/LazyPPL programs.
    Falls back to Python AST parsing for PyMC programs.
    """

    def __init__(
        self,
        lazyppl_dir: str | None = None,
        timeout_s: float = COMPILATION_TIMEOUT_S,
    ):
        self.lazyppl_dir = lazyppl_dir
        self.timeout_s = timeout_s
        self._lazyppl_executor: Any = None

    def _get_lazyppl_executor(self) -> Any:
        if self._lazyppl_executor is None:
            from synthstats.executors.lazyppl_sandbox import LazyPPLExecutor

            self._lazyppl_executor = LazyPPLExecutor(
                lazyppl_dir=self.lazyppl_dir, compile_only=True
            )
        return self._lazyppl_executor

    def _check_lazyppl(self, code: str) -> tuple[bool, str | None]:
        from synthstats.core.types import ToolCall

        executor = self._get_lazyppl_executor()
        payload = ToolCall(name="lazyppl", input={"code": code}, raw="")
        result = executor.execute(payload, timeout_s=self.timeout_s)
        return result.success, result.error

    def _check_python(self, code: str) -> tuple[bool, str | None]:
        import ast

        try:
            ast.parse(code)
            compile(code, "<submission>", "exec")
            return True, None
        except SyntaxError as exc:
            return False, str(exc)

    def score(self, *, task_name: str, trajectory: Trajectory, artifacts: dict[str, Any]) -> Reward:
        program = artifacts.get("program", "")
        raw_language = artifacts.get("language")
        language = normalize_runtime_name(raw_language, default=DEFAULT_RUNTIME_NAME)
        language = language or str(raw_language or DEFAULT_RUNTIME_NAME)
        runtime_spec = get_runtime_spec(language) if language in {"pymc", "lazyppl"} else None

        if not program.strip():
            return Reward(
                total=0.0,
                components={"compiles": 0.0},
                info=annotate_reward_info(
                    {"language": language, "error": "empty_program"},
                    source="validation.compilation",
                ),
            )

        if runtime_spec is not None and runtime_spec.execution_backend == "haskell":
            compiles, error_msg = self._check_lazyppl(program)
        else:
            compiles, error_msg = self._check_python(program)

        info: dict[str, Any] = {"language": language}
        if error_msg is not None:
            info["error"] = error_msg
            logger.info("[diag] compilation failed (%s): %s", language, error_msg)

        return Reward(
            total=1.0 if compiles else 0.0,
            components={"compiles": 1.0 if compiles else 0.0},
            info=annotate_reward_info(info, source="validation.compilation"),
        )
