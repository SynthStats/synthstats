from __future__ import annotations

import json

import compute_atif_rewards
import compute_pilot_elpd
import generate_pilot_programs
import pytest
import sft_to_atif
from harbor.models.trajectories.trajectory import Trajectory

from synthstats.executors.lazyppl_sandbox import LazyPPLExecutor
from synthstats.integrations.harbor.collector import HarborCollector


class ByteTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))


@pytest.mark.parametrize("rescorer", ["compilation", "pilot"])
def test_rescoring_exported_record_changes_collected_reward(tmp_path, rescorer):
    record = sft_to_atif.build_atif_record(
        0,
        "Write a model.",
        '<submit_program language="haskell"></submit_program>',
        ByteTokenizer(),
        record={"validation_json": json.dumps({"ok": True})},
    )
    path = tmp_path / (
        "sft-lazyppl-0000.json" if rescorer == "compilation" else "pilot-dugongs-0000.json"
    )
    path.write_text(json.dumps(record))
    collector = HarborCollector(tmp_path, shuffle=False)
    assert collector.collect_atif()[0][1] == 1.0

    if rescorer == "compilation":
        status = compute_atif_rewards.main(atif_dir=tmp_path)
    else:
        status = compute_pilot_elpd.main(
            atif_dir=tmp_path, executor=LazyPPLExecutor(lazyppl_dir=str(tmp_path))
        )
    assert status == 1
    rescored = json.loads(path.read_text())
    collected = collector.collect_atif()[0][1]
    assert collected == 1e-8
    validated = Trajectory.model_validate(rescored)
    assert validated.final_metrics.extra["reward_source"] == "verifier_missing_program"
    assert validated.final_metrics.extra["evidence_kind"] == "verifier"
    assert validated.final_metrics.extra["truth_owner"] == "verifier"
    assert "reward" not in rescored["final_metrics"]


def test_pilot_export_uses_harbor_metrics_extension(tmp_path):
    record = generate_pilot_programs.build_atif(
        env_name="dugongs",
        idx=0,
        system_msg="Write a probabilistic program.",
        user_msg="Model the observations.",
        completion="module Submission where",
        code="module Submission where",
        token_ids=[1, 2],
        logprobs=[-0.1, -0.2],
        compiles=True,
        reward_source="validation.compilation",
    )
    validated = Trajectory.model_validate(record)
    assert validated.final_metrics.extra["reward"] == 1.0
    assert validated.final_metrics.extra["reward_source"] == "validation.compilation"
    (tmp_path / "pilot-dugongs-0000.json").write_text(json.dumps(record))
    assert HarborCollector(tmp_path, shuffle=False).collect_atif()[0][1] == 1.0


def test_collector_rejects_reward_outside_extension(tmp_path):
    record = sft_to_atif.build_atif_record(
        0,
        "Write a model.",
        '<submit_program language="haskell">module Submission where</submit_program>',
        ByteTokenizer(),
    )
    record["final_metrics"]["reward"] = 0.5
    (tmp_path / "sft-lazyppl-0000.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="final_metrics.extra"):
        HarborCollector(tmp_path, shuffle=False).collect_atif()
