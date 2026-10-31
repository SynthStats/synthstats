from __future__ import annotations

import pytest

from synthstats.core import runtime_registry


@pytest.mark.parametrize(
    ("runtime_name", "alias", "metadata_key", "default_version", "artifact_key", "task_prefix"),
    [
        ("pymc", "pymc", "pymc_version", "v5", "elpd", "pymc"),
        ("lazyppl", "haskell", "lazyppl_version", "lazyppl-1.0", "log_lik_matrix", "lazyppl"),
    ],
)
def test_runtime_registry_parity_matrix(
    runtime_name: str,
    alias: str,
    metadata_key: str,
    default_version: str,
    artifact_key: str,
    task_prefix: str,
) -> None:
    spec = runtime_registry.get_runtime_spec(alias)

    assert runtime_registry.normalize_runtime_name(alias) == runtime_name
    assert spec.name == runtime_name
    assert spec.metadata_version_key == metadata_key
    assert spec.default_version == default_version
    assert spec.score_artifact_key == artifact_key
    assert (
        runtime_registry.runtime_name_from_metadata({metadata_key: default_version}) == runtime_name
    )
    assert runtime_registry.runtime_name_from_payload({"language": alias}) == runtime_name
    assert (
        runtime_registry.runtime_name_from_payload(
            {"shared_trial_record": {"runtime": {"name": alias}}}
        )
        == runtime_name
    )
    assert (
        runtime_registry.score_artifact_key_from_payload({metadata_key: default_version})
        == artifact_key
    )
    assert runtime_registry.build_harbor_task_id(alias, "gp_regression", sequence=7).startswith(
        f"{task_prefix}-gp-regression-"
    )
