"""Shared Harbor reward artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from synthstats.core.reward_semantics import annotate_reward_info
from synthstats.core.types import Reward

REWARD_JSON_NAME = "reward.json"
REWARD_ARTIFACT_JSON_NAME = "reward_artifact.json"
REWARD_TXT_NAME = "reward.txt"
REWARD_ARTIFACT_SCHEMA_VERSION = "synthstats.harbor.reward.v1"

_JSON_RESERVED_KEYS = frozenset(
    {
        "schema_version",
        "reward",
        "reward_source",
        "evidence_kind",
        "truth_owner",
        "loo_status",
        "diagnostics",
        "provenance",
        "components",
        "info",
    }
)
_INFO_TOP_LEVEL_KEYS = frozenset(
    {
        "reward_source",
        "evidence_kind",
        "truth_owner",
    }
)


def _coerce_float_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return out


def build_reward_artifact(
    reward: Reward,
    *,
    reward_source: str,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical structured verifier artifact."""
    reward_info = annotate_reward_info(
        reward.info,
        reward_source=reward_source,
    )
    artifact_info = {
        key: value for key, value in reward_info.items() if key not in _INFO_TOP_LEVEL_KEYS
    }
    diagnostics = reward.info.get("diagnostics", {})
    loo_status = reward.info.get("loo_status", "not_computed")
    components = dict(reward.components)
    elpd = components.get("elpd")

    return {
        "schema_version": REWARD_ARTIFACT_SCHEMA_VERSION,
        "reward": float(reward.total),
        "reward_source": reward_source,
        "evidence_kind": reward_info["evidence_kind"],
        "truth_owner": reward_info["truth_owner"],
        "elpd": float(elpd) if isinstance(elpd, int | float) else None,
        "loo_status": str(loo_status),
        "diagnostics": diagnostics if isinstance(diagnostics, dict) else {},
        "provenance": provenance if isinstance(provenance, dict) else {},
        "components": components,
        "info": artifact_info,
    }


def write_reward_artifacts(
    artifact: dict[str, Any],
    *,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write Harbor's numeric reward and the detailed verifier companion."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    reward_total = float(artifact.get("reward", 0.0))
    reward_json_path = directory / REWARD_JSON_NAME
    reward_txt_path = directory / REWARD_TXT_NAME

    reward_json_path.write_text(json.dumps({"reward": reward_total}, indent=2))
    (directory / REWARD_ARTIFACT_JSON_NAME).write_text(
        json.dumps(artifact, indent=2, sort_keys=True)
    )
    reward_txt_path.write_text(f"{reward_total}\n")
    return reward_json_path, reward_txt_path


def reward_artifact_to_reward(data: dict[str, Any], *, path: str) -> Reward:
    """Convert a structured or plain reward JSON payload into ``Reward``."""
    if (
        "schema_version" in data
        or "components" in data
        or "loo_status" in data
        or any(key in data for key in ("reward_source", "evidence_kind", "truth_owner"))
    ):
        extra_fields = {key: value for key, value in data.items() if key not in _JSON_RESERVED_KEYS}
        components = _coerce_float_map(data.get("components"))
        for key, value in extra_fields.items():
            if isinstance(value, int | float):
                components.setdefault(key, float(value))
        elpd = data.get("elpd")
        if isinstance(elpd, int | float) and "elpd" not in components:
            components["elpd"] = float(elpd)

        extra_info = data.get("info")
        info = annotate_reward_info(
            {
                "path": path,
                "schema_version": data.get("schema_version"),
                "loo_status": data.get("loo_status", "not_computed"),
                "diagnostics": data.get("diagnostics", {}),
                "provenance": data.get("provenance", {}),
                **extra_fields,
                **(extra_info if isinstance(extra_info, dict) else {}),
            },
            reward_source=str(data.get("reward_source", "unknown")),
            source="reward_json",
        )
        info["source"] = "reward_json"
        info["evidence_kind"] = str(data.get("evidence_kind", info["evidence_kind"]))
        info["truth_owner"] = str(data.get("truth_owner", info["truth_owner"]))

        return Reward(
            total=float(data.get("reward", 0.0)),
            components=components,
            info=info,
        )

    return Reward(
        total=float(data.get("reward", 0.0)),
        components={
            key: float(value)
            for key, value in data.items()
            if key not in _JSON_RESERVED_KEYS and isinstance(value, int | float)
        },
        info=annotate_reward_info(
            {"path": path},
            reward_source="sidecar",
            source="sidecar",
        ),
    )
