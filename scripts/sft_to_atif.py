#!/usr/bin/env python3
"""Convert SFT-style records to ATIF v1.6 JSON trajectory files.

Reads: the JSONL file supplied with --input-path, containing SFT or richer dataset rows
Writes: data/atif_trajectories/sft-<runtime>-{0000..}.json + manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict

from _utils import (
    ATIF_SCHEMA_VERSION,
    build_atif_agent,
    write_json_atomic,
)

from synthstats.core import platform_adapters
from synthstats.core.reward_semantics import annotate_reward_info
from synthstats.core.runtime_registry import normalize_runtime_name, runtime_name_from_payload

log = logging.getLogger(__name__)

CODE_FENCE_PATTERN = re.compile(r"```([A-Za-z0-9_+-]*)\s*(.*?)```", re.DOTALL)
SUBMIT_PROGRAM_PATTERN = re.compile(
    r'<submit_program(?:\s+language="([^"]*)")?>(.*?)</submit_program>',
    re.DOTALL,
)

SYSTEM_PROMPT = (
    "You are an expert probabilistic programmer. "
    "Given the scientific question and data summary, write a Bayesian model for the "
    "{runtime} runtime. Follow the task's program requirements."
)

MODEL_NAME = "sft-data"


class TokenizerLike(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


class SFTRecord(TypedDict):
    prompt: str
    completion: str


class ManifestEntry(TypedDict):
    session_id: str
    file: str
    reward: float
    reward_source: str
    evidence_kind: str
    truth_owner: str


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_output_dir() -> Path:
    return project_root() / "data" / "atif_trajectories"


def load_tokenizer() -> TokenizerLike:
    """Load the Qwen3.5-0.8B tokenizer."""
    try:
        from transformers import AutoTokenizer

        log.info("Loading Qwen/Qwen3.5-0.8B tokenizer...")
        t0 = time.monotonic()
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
        log.info("Tokenizer loaded in %.1fs", time.monotonic() - t0)
        return tok
    except Exception as exc:
        raise RuntimeError("Cannot load the required Qwen3.5-0.8B tokenizer") from exc


def _string(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _json_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def extract_think(completion: str) -> str | None:
    """Extract content between <think>...</think> tags."""
    m = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
    return m.group(1).strip() if m else None


def _dataset_adapters():
    return platform_adapters


def _runtime_name(record: Mapping[str, Any] | None) -> str | None:
    if record is None:
        return None
    if "id" in record:
        adapters = _dataset_adapters()
        try:
            runtime = adapters.runtime_descriptor_from_dataset_row(record)
        except Exception:
            runtime = None
        if runtime is not None and isinstance(runtime.name, str) and runtime.name:
            return runtime.name
    return runtime_name_from_payload(record)


def _normalize_runtime_name(name: str | None, *, default: str | None = None) -> str | None:
    return normalize_runtime_name(name, default=default)


def extract_code(
    completion: str,
    *,
    default_language: str | None = None,
) -> tuple[str, str]:
    """Extract language and code from submit tags, JSON tool calls, or fenced code."""
    submit_match = SUBMIT_PROGRAM_PATTERN.search(completion)
    if submit_match:
        language = submit_match.group(1)
        return (language or default_language or "unknown").strip(), submit_match.group(2).strip()

    try:
        payload = json.loads(completion)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict) and payload.get("name") in {"submit_program", "submit"}:
        arguments = payload.get("arguments", {})
        if isinstance(arguments, dict):
            code = arguments.get("code") or arguments.get("program") or arguments.get("content")
            if isinstance(code, str) and code.strip():
                language = arguments.get("language")
                return (str(language or default_language or "unknown")).strip(), code.strip()

    fence_match = CODE_FENCE_PATTERN.search(completion)
    if fence_match:
        fence_language = default_language or fence_match.group(1).strip() or "unknown"
        return fence_language, fence_match.group(2).strip()

    raise ValueError("No submitted program found in completion")


def compute_heuristic_reward(code: str, *, runtime_name: str | None = None) -> float:
    """Heuristic reward for dataset rows without verifier-backed validation."""
    normalized_runtime = (runtime_name or "").strip().lower()
    if normalized_runtime == "pymc":
        markers = ("import pymc", "from pymc", "pm.Model", "pymc.Model", "pm.sample")
        return 1.0 if any(marker in code for marker in markers) else 0.0
    return 1.0 if "import LazyPPL" in code else 0.0


def tokenize_text(tokenizer: TokenizerLike | None, text: str) -> list[int]:
    """Encode text with the supplied model tokenizer."""
    if tokenizer is None:
        raise ValueError("A model tokenizer is required to produce token IDs")
    return tokenizer.encode(text, add_special_tokens=False)


def _build_foundry_metadata(
    record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if record is None:
        return {}

    dataset_identity = {
        key: value
        for key in ("id", "prompt_id", "env_id", "split", "trace_id", "env_name", "data_ref")
        if (value := record.get(key))
    }
    metadata: dict[str, Any] = {
        "source": "synthstats.dataset_row",
        "dataset_row": dataset_identity,
    }

    if "id" in record:
        adapters = _dataset_adapters()

        try:
            generation = adapters.generation_record_from_dataset_row(record)
            metadata["shared_generation_record"] = generation.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001 - best-effort enrichment
            metadata["shared_generation_error"] = str(exc)
        try:
            trial = adapters.trial_record_from_dataset_row(record)
            metadata["shared_trial_record"] = trial.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001 - best-effort enrichment
            metadata["shared_trial_error"] = str(exc)
        try:
            trace = adapters.trace_record_from_dataset_row(record)
            if trace is not None:
                metadata["shared_trace_record"] = trace.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001 - best-effort enrichment
            metadata["shared_trace_error"] = str(exc)

    return metadata


def _build_final_metrics(
    *,
    code: str,
    record: Mapping[str, Any] | None,
    runtime_name: str | None,
) -> dict[str, Any]:
    validation = _json_dict(record.get("validation_json")) if record is not None else {}
    semantic_score = _numeric(validation.get("semantic_score"))
    if semantic_score is not None:
        reward = semantic_score
        reward_source = "validation.semantic_score"
    elif isinstance(validation.get("ok"), bool):
        reward = 1.0 if validation["ok"] else 0.0
        reward_source = "validation.ok"
    else:
        reward = compute_heuristic_reward(code, runtime_name=runtime_name)
        reward_source = "heuristic.runtime_import"

    final_metrics_extra = annotate_reward_info(
        {
            "reward": reward,
            "reward_source": reward_source,
        },
        reward_source=reward_source,
    )
    for key in (
        "ok",
        "message",
        "semantic_metric",
        "semantic_score",
        "semantic_status",
        "evidence_kind",
        "truth_owner",
    ):
        if key in validation:
            final_metrics_extra[key] = validation[key]
    return {"extra": final_metrics_extra}


def validate_atif_record(atif: dict[str, Any]) -> None:
    """Validate an exported or rescored trajectory with Harbor's ATIF schema."""
    from harbor.models.trajectories.trajectory import Trajectory

    Trajectory.model_validate(atif)


def build_atif_record(
    index: int,
    prompt: str,
    completion: str,
    tokenizer: TokenizerLike | None,
    *,
    record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a single ATIF v1.6 trajectory dict."""
    reasoning = extract_think(completion)
    runtime_name = _runtime_name(record)
    language, code = extract_code(completion, default_language=runtime_name)
    session_runtime_name = (
        _normalize_runtime_name(
            runtime_name or language,
            default="pymc",
        )
        or "pymc"
    )

    prompt_token_ids = tokenize_text(tokenizer, prompt)
    completion_token_ids = tokenize_text(tokenizer, completion)
    n_completion = len(completion_token_ids)

    session_id = f"sft-{session_runtime_name}-{index:04d}"

    agent_step: dict = {
        "step_id": 3,
        "source": "agent",
        "message": completion,
        "tool_calls": [
            {
                "tool_call_id": "tc_001",
                "function_name": "submit_program",
                "arguments": {"language": language, "code": code},
            }
        ],
        "metrics": {
            "prompt_tokens": len(prompt_token_ids),
            "completion_tokens": n_completion,
            "completion_token_ids": completion_token_ids,
        },
    }
    if reasoning is not None:
        agent_step["reasoning_content"] = reasoning

    atif = {
        "schema_version": ATIF_SCHEMA_VERSION,
        "session_id": session_id,
        "agent": build_atif_agent(MODEL_NAME),
        "steps": [
            {
                "step_id": 1,
                "source": "system",
                "message": SYSTEM_PROMPT.format(runtime=session_runtime_name),
            },
            {"step_id": 2, "source": "user", "message": prompt},
            agent_step,
        ],
        "final_metrics": _build_final_metrics(
            code=code,
            record=record,
            runtime_name=session_runtime_name,
        ),
    }
    metadata = _build_foundry_metadata(record)
    metadata["training_data_kind"] = "supervised"
    metadata["behavior_logprobs_available"] = False
    if metadata:
        atif["extra"] = metadata
    validate_atif_record(atif)
    return atif


def load_sft_records(input_path: Path) -> tuple[list[tuple[int, SFTRecord]], int]:
    """Load line-delimited JSON SFT records from disk.

    Returns valid records paired with their original non-empty record index, plus a
    count of malformed JSON lines that were skipped.
    """
    records: list[tuple[int, SFTRecord]] = []
    parse_errors = 0
    record_index = 0
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append((record_index, json.loads(line)))
            except json.JSONDecodeError as exc:
                log.error("Input line %d is invalid JSON: %s", line_number, exc.msg)
                parse_errors += 1
            record_index += 1
    return records, parse_errors


def build_manifest_entry(atif: dict[str, Any]) -> ManifestEntry:
    metrics_payload = atif["final_metrics"]["extra"]
    return {
        "session_id": atif["session_id"],
        "file": f"{atif['session_id']}.json",
        "reward": metrics_payload["reward"],
        "reward_source": str(metrics_payload["reward_source"]),
        "evidence_kind": str(metrics_payload["evidence_kind"]),
        "truth_owner": str(metrics_payload["truth_owner"]),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Convert SFT JSONL rows or dataset rows into ATIF trajectory files."),
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help=("Path to the source JSONL file. Accepts plain SFT rows or dataset rows."),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for ATIF JSON files and manifest.json.",
    )
    parser.add_argument(
        "--tokenizer",
        choices=("auto",),
        default="auto",
        help="Load the Qwen3.5-0.8B tokenizer.",
    )
    return parser


def main(
    *,
    input_path: Path,
    output_dir: Path | None = None,
    tokenizer: TokenizerLike | None | Literal["auto"] = "auto",
) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    resolved_output_dir = output_dir or default_output_dir()

    if not input_path.exists():
        log.error("Input not found: %s", input_path)
        return 1

    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_tokenizer = load_tokenizer() if tokenizer == "auto" else tokenizer
    records, errors = load_sft_records(input_path)

    log.info("Loaded %d SFT records", len(records))

    manifest_entries: list[ManifestEntry] = []

    for record_index, rec in records:
        try:
            prompt = str(rec["prompt"])
            completion = str(rec["completion"])
            atif = build_atif_record(
                record_index,
                prompt,
                completion,
                resolved_tokenizer,
                record=rec,
            )
        except Exception as exc:
            log.error("Record %d failed: %s", record_index, exc)
            errors += 1
            continue

        manifest_entry = build_manifest_entry(atif)
        filename = manifest_entry["file"]
        out_path = resolved_output_dir / filename
        write_json_atomic(out_path, atif)
        manifest_entries.append(manifest_entry)

    manifest_path = resolved_output_dir / "manifest.json"
    write_json_atomic(
        manifest_path,
        {
            "schema_version": ATIF_SCHEMA_VERSION,
            "count": len(manifest_entries),
            "files": manifest_entries,
        },
    )

    log.info(
        "Wrote %d ATIF files to %s (%d errors)",
        len(manifest_entries),
        resolved_output_dir,
        errors,
    )
    return 0 if errors == 0 else 1


def cli() -> int:
    args = build_arg_parser().parse_args()
    tokenizer: TokenizerLike | None | Literal["auto"]
    tokenizer = "auto" if args.tokenizer == "auto" else None
    return main(
        input_path=args.input_path,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    raise SystemExit(cli())
