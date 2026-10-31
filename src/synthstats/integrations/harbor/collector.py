"""Harbor trajectory collector.

Reads pre-collected ATIF trajectory JSON files from a directory.
Two collection modes:
- collect(): returns CollectedTrajectory objects
- collect_atif(): returns raw (atif_dict, reward) pairs

No Harbor SDK required for offline reading.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from synthstats.core.reward_semantics import (
    annotate_reward_info,
    reward_is_verifier_owned,
)
from synthstats.core.types import Reward
from synthstats.integrations.harbor.converter import (
    atif_to_trajectory,
    extract_program,
    read_reward_sidecar,
    validate_atif_schema,
)
from synthstats.integrations.harbor.generator import extract_lineage_artifacts_from_atif
from synthstats.integrations.harbor.layout import reward_directory
from synthstats.train.data.collectors import CollectedTrajectory

if TYPE_CHECKING:
    from synthstats.core.judge import Judge

logger = logging.getLogger(__name__)

_STALE_VERIFICATION_SOURCE = "heuristic.code_change"
_STALE_VERIFICATION_REASON = "program_mismatch"


class HarborCollector:
    """Reads ATIF trajectory files and produces CollectedTrajectory batches."""

    def __init__(
        self,
        atif_dir: str | Path,
        *,
        judge: Judge | None = None,
        prefer_verifier_reward: bool = True,
        tokenizer: Any | None = None,
        shuffle: bool = True,
        cycle: bool = True,
        task_name: str = "harbor",
    ) -> None:
        self.atif_dir = Path(atif_dir)
        if not self.atif_dir.is_dir():
            raise FileNotFoundError(f"ATIF directory not found: {self.atif_dir}")

        self.judge = judge
        self.prefer_verifier_reward = prefer_verifier_reward
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.cycle = cycle
        self.task_name = task_name

        self._files = self._scan_files()
        if not self._files:
            raise FileNotFoundError(f"no ATIF JSON files in {self.atif_dir}")

        self._index = 0
        self._epoch = 0
        self._last_collection_metrics = self._empty_collection_metrics()
        logger.info("HarborCollector: %d ATIF files in %s", len(self._files), self.atif_dir)

    def _scan_files(self) -> list[Path]:
        """Find all trajectory JSON files in the directory."""
        files = sorted(
            p
            for p in self.atif_dir.rglob("*.json")
            if p.stem
            not in (
                "reward",
                "reward_artifact",
                "synthstats_result",
                "metadata",
                "task",
                "manifest",
            )
        )
        if self.shuffle:
            random.shuffle(files)
        return files

    def _next_file(self) -> Path | None:
        if self._index >= len(self._files):
            if not self.cycle:
                return None
            self._epoch += 1
            self._index = 0
            if self.shuffle:
                random.shuffle(self._files)
        if self._index >= len(self._files):
            return None
        path = self._files[self._index]
        self._index += 1
        return path

    def _load_atif(self, path: Path) -> dict[str, Any] | None:
        try:
            with open(path) as f:
                atif = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("skipping %s: %s", path, e)
            return None

        ok, err = validate_atif_schema(atif)
        if not ok:
            logger.warning("skipping %s: %s", path, err)
            return None
        return atif

    def _annotate_reward_semantics(self, reward: Reward, *, reward_source: str) -> Reward:
        info = annotate_reward_info(
            reward.info,
            reward_source=reward_source,
        )
        return Reward(
            total=reward.total,
            components=dict(reward.components),
            info=info,
            factors=dict(reward.factors),
            scalarization=reward.scalarization,
        )

    def _trajectory_artifacts_from_atif(
        self,
        atif: dict[str, Any],
        *,
        atif_path: Path | None = None,
    ) -> dict[str, Any]:
        artifacts: dict[str, Any] = {
            "session_id": atif.get("session_id"),
            "schema_version": atif.get("schema_version"),
        }

        final_metrics = self._normalized_final_metrics(atif, atif_path=atif_path)
        if final_metrics is not None:
            artifacts["final_metrics"] = final_metrics

        metadata = atif.get("extra", atif.get("metadata", {}))
        if isinstance(metadata, Mapping):
            artifacts["atif_metadata"] = dict(metadata)

        # shared lineage extraction (same logic as generator)
        artifacts.update(extract_lineage_artifacts_from_atif(atif))

        return {key: value for key, value in artifacts.items() if value is not None}

    @staticmethod
    def _final_metrics_reward_source(
        final_metrics: Mapping[str, Any],
    ) -> str | None:
        reward_source = final_metrics.get("reward_source")
        if isinstance(reward_source, str) and reward_source.strip():
            return reward_source
        if "semantic_score" in final_metrics:
            return "validation.semantic_score"
        if isinstance(final_metrics.get("ok"), bool):
            return "validation.ok"
        if any(key in final_metrics for key in ("elpd", "elpd_status", "log_lik_shape")):
            if final_metrics.get("elpd_status") == "missing_code":
                return "verifier_missing_program"
            return "validation.elpd"
        return None

    @staticmethod
    def _strip_provenance_fields(final_metrics: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in final_metrics.items()
            if key
            not in {
                "source",
                "reward_source",
                "evidence_kind",
                "truth_owner",
                "verification_stale",
                "verification_stale_reason",
            }
        }

    @staticmethod
    def _flatten_final_metrics(final_metrics: Mapping[str, Any]) -> dict[str, Any]:
        if "reward" in final_metrics:
            raise ValueError("ATIF rewards must be stored in final_metrics.extra")
        extra = final_metrics.get("extra")
        return dict(extra) if isinstance(extra, Mapping) else {}

    def _stale_verification_source(
        self,
        atif_path: Path | None,
        atif: Mapping[str, Any],
    ) -> str | None:
        if atif_path is None:
            return None

        backup_path = atif_path.with_suffix(f"{atif_path.suffix}.bak")
        if not backup_path.is_file():
            return None

        raw_current = atif.get("final_metrics", {})
        if not isinstance(raw_current, Mapping):
            return None
        current_metrics = self._flatten_final_metrics(raw_current)
        if "reward" not in current_metrics:
            return None

        backup_atif = self._load_atif(backup_path)
        if backup_atif is None:
            return None

        raw_backup = backup_atif.get("final_metrics", {})
        if not isinstance(raw_backup, Mapping):
            return None
        backup_metrics = self._flatten_final_metrics(raw_backup)

        current_code = extract_program(dict(atif))
        backup_code = extract_program(backup_atif)
        if current_code is None or backup_code is None or current_code == backup_code:
            return None

        if self._strip_provenance_fields(current_metrics) != self._strip_provenance_fields(
            backup_metrics
        ):
            return None

        return _STALE_VERIFICATION_SOURCE

    def _normalized_final_metrics(
        self,
        atif: Mapping[str, Any],
        *,
        atif_path: Path | None = None,
    ) -> dict[str, Any] | None:
        final_metrics = atif.get("final_metrics", {})
        if not isinstance(final_metrics, Mapping):
            return None

        normalized = self._flatten_final_metrics(final_metrics)
        stale_reward_source = self._stale_verification_source(atif_path, atif)
        if stale_reward_source is not None:
            annotated = annotate_reward_info(
                normalized,
                reward_source=stale_reward_source,
                source="final_metrics",
            )
            annotated["source"] = "final_metrics"
            annotated["reward_source"] = _STALE_VERIFICATION_SOURCE
            annotated["evidence_kind"] = "heuristic"
            annotated["truth_owner"] = "heuristic"
            annotated["verification_stale"] = True
            annotated["verification_stale_reason"] = _STALE_VERIFICATION_REASON
            return annotated

        reward_source = self._final_metrics_reward_source(normalized)
        if reward_source is None:
            return normalized

        annotated = annotate_reward_info(
            normalized,
            reward_source=reward_source,
            source="final_metrics",
        )
        annotated["source"] = "final_metrics"
        return annotated

    @staticmethod
    def _empty_collection_metrics() -> dict[str, int]:
        return {
            "episodes_collected": 0,
            "episodes_with_trace_id": 0,
            "episodes_with_task_id": 0,
            "episodes_with_dataset_row": 0,
            "episodes_with_shared_generation_record": 0,
            "episodes_with_shared_trial_record": 0,
            "episodes_with_shared_trace_record": 0,
            "episodes_with_verifier_reward": 0,
        }

    def _record_collection_metrics(
        self,
        metrics: dict[str, int],
        *,
        artifacts: Mapping[str, Any],
        reward: Reward,
    ) -> None:
        metrics["episodes_collected"] += 1
        if isinstance(artifacts.get("trace_id"), str) and artifacts["trace_id"]:
            metrics["episodes_with_trace_id"] += 1
        if isinstance(artifacts.get("task_id"), str) and artifacts["task_id"]:
            metrics["episodes_with_task_id"] += 1
        if isinstance(artifacts.get("dataset_row"), Mapping):
            metrics["episodes_with_dataset_row"] += 1
        for key in (
            "shared_generation_record",
            "shared_trial_record",
            "shared_trace_record",
        ):
            if isinstance(artifacts.get(key), Mapping):
                metrics[f"episodes_with_{key}"] += 1
        if reward_is_verifier_owned(reward):
            metrics["episodes_with_verifier_reward"] += 1

    def _judge_reward(self, atif: dict[str, Any]) -> Reward | None:
        if self.judge is None:
            return None

        program = extract_program(atif)
        dummy_reward = Reward(total=0.0, components={}, info={})
        traj = atif_to_trajectory(atif, dummy_reward, tokenizer=self.tokenizer)
        artifacts: dict[str, Any] = {}
        if program:
            artifacts["program"] = program
            agent_cfg = atif.get("agent", {})
            artifacts["language"] = agent_cfg.get("language", "lazyppl")
        reward = self.judge.score(
            task_name=self.task_name,
            trajectory=traj,
            artifacts=artifacts,
        )
        return self._annotate_reward_semantics(reward, reward_source="judge")

    def _sidecar_rewards(self, task_dir: Path) -> tuple[Reward | None, Reward | None]:
        verifier_reward: Reward | None = None
        fallback_reward: Reward | None = None
        for name in ("reward_artifact.json", "reward.json", "reward.txt"):
            reward = read_reward_sidecar(str(task_dir / name))
            if reward is not None:
                if reward_is_verifier_owned(reward):
                    verifier_reward = reward
                    break
                if fallback_reward is None:
                    fallback_reward = reward
        return verifier_reward, fallback_reward

    def _final_metrics_reward(
        self,
        atif: dict[str, Any],
        *,
        atif_path: Path | None = None,
    ) -> Reward | None:
        final = self._normalized_final_metrics(atif, atif_path=atif_path)
        if final is None:
            return None
        if "reward" not in final:
            return None
        reward_total = float(final["reward"])
        info = {key: value for key, value in final.items() if key != "reward"}
        return Reward(
            total=reward_total,
            components={"raw": reward_total},
            info=annotate_reward_info(
                info,
                reward_source=str(final.get("reward_source") or "final_metrics"),
                source="final_metrics",
            ),
        )

    def _log_reward_mismatch(
        self,
        *,
        atif_path: Path,
        verifier_reward: Reward | None,
        final_metrics_reward: Reward | None,
    ) -> None:
        if verifier_reward is None or final_metrics_reward is None:
            return
        if verifier_reward.total == final_metrics_reward.total:
            return
        logger.warning(
            "reward source mismatch for %s: verifier=%s final_metrics=%s",
            atif_path,
            verifier_reward.total,
            final_metrics_reward.total,
        )

    def _resolve_reward(self, atif: dict[str, Any], atif_path: Path) -> Reward:
        """Resolve reward with verifier truth first and heuristic fallbacks second."""
        task_dir = reward_directory(atif_path)
        verifier_reward, sidecar_reward = self._sidecar_rewards(task_dir)
        final_metrics_reward = self._final_metrics_reward(atif, atif_path=atif_path)
        judge_reward = self._judge_reward(atif) if self.judge is not None else None

        self._log_reward_mismatch(
            atif_path=atif_path,
            verifier_reward=verifier_reward,
            final_metrics_reward=final_metrics_reward,
        )

        if self.prefer_verifier_reward:
            for reward in (verifier_reward, sidecar_reward, final_metrics_reward, judge_reward):
                if reward is not None:
                    return reward
        else:
            for reward in (judge_reward, verifier_reward, sidecar_reward, final_metrics_reward):
                if reward is not None:
                    return reward

        logger.warning("no reward source for %s, defaulting to 0.0", atif_path)
        return Reward(
            total=0.0,
            components={},
            info=annotate_reward_info(
                {},
                reward_source="default",
            ),
        )

    def collect_atif(
        self,
        episodes: int = 1,
    ) -> list[tuple[dict[str, Any], float]]:
        """Read ATIF files and return raw (atif_dict, reward) pairs.

        This is the ATIF-native path, without Trajectory or CollectedTrajectory
        intermediate types.
        """
        results: list[tuple[dict[str, Any], float]] = []
        metrics = self._empty_collection_metrics()

        for _ in range(episodes):
            path = self._next_file()
            if path is None:
                break

            atif = self._load_atif(path)
            if atif is None:
                continue

            reward = self._resolve_reward(atif, path)
            self._record_collection_metrics(
                metrics,
                artifacts=self._trajectory_artifacts_from_atif(atif, atif_path=path),
                reward=reward,
            )
            results.append((atif, reward.total))

        self._last_collection_metrics = metrics
        return results

    def collect(
        self,
        episodes: int = 1,
        *,
        temperature: float = 1.0,
        compute_ref_log_probs: bool = False,
    ) -> list[CollectedTrajectory]:
        """Read and convert `episodes` ATIF files into CollectedTrajectory."""
        collected: list[CollectedTrajectory] = []
        metrics = self._empty_collection_metrics()

        for _ in range(episodes):
            path = self._next_file()
            if path is None:
                break

            atif = self._load_atif(path)
            if atif is None:
                continue

            reward = self._resolve_reward(atif, path)
            traj = atif_to_trajectory(atif, reward, tokenizer=self.tokenizer)

            # build log_probs tensor from ATIF logprobs (one scalar per generation)
            n_gens = len(traj.token_logprobs)
            if n_gens > 0:
                # sum log_probs per generation for action-level training
                log_probs = torch.tensor(
                    [sum(lps) for lps in traj.token_logprobs], dtype=torch.float32
                )
                entropy = torch.zeros(n_gens, dtype=torch.float32)
            else:
                logger.warning("skipping zero-generation ATIF episode from %s", path)
                continue

            # extract observations and actions from messages
            observations: list[str] = []
            actions: list[dict[str, Any]] = []
            for msg in traj.messages:
                if msg.role in ("system", "user", "tool"):
                    observations.append(msg.content)
                elif msg.role == "assistant":
                    actions.append({"type": "generation", "content": msg.content})

            artifacts = self._trajectory_artifacts_from_atif(atif, atif_path=path)
            self._record_collection_metrics(metrics, artifacts=artifacts, reward=reward)

            collected.append(
                CollectedTrajectory(
                    observations=observations,
                    actions=actions,
                    log_probs=log_probs,
                    entropy=entropy,
                    reward=reward.total,
                    temperature=temperature,
                    eos_logprobs=None,
                    prompts=None,
                    completions=None,
                    artifacts=artifacts,
                )
            )

        self._last_collection_metrics = metrics
        return collected

    @property
    def num_files(self) -> int:
        return len(self._files)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def last_collection_metrics(self) -> dict[str, int]:
        return dict(self._last_collection_metrics)
