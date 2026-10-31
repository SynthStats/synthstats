from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_FILES = [REPO_ROOT / "README.md"]
SCRIPT_FILES = [
    REPO_ROOT / "scripts" / "launch-training.sh",
]
RUNNER_CONFIG_FILES = [
    REPO_ROOT / "src" / "synthstats" / "configs" / "runner" / "skyrl.yaml",
    REPO_ROOT / "src" / "synthstats" / "configs" / "runner" / "skyrl_harbor.yaml",
]
ENV_CONFIG_FILES = [
    REPO_ROOT / "src" / "synthstats" / "configs" / "env" / "_statistical_modeling_base.yaml",
    REPO_ROOT / "src" / "synthstats" / "configs" / "env" / "_lazyppl_base.yaml",
]
TASK_CONFIG_FILES = [
    REPO_ROOT / "src" / "synthstats" / "configs" / "task" / "statistical_modeling.yaml",
]
TRAIN_CONFIG_FILE = REPO_ROOT / "src" / "synthstats" / "configs" / "train.yaml"
INVALID_RUNNER_TOKENS = ("runner=tinker", "runner=local", "runner=harbor", "runner=runpod")
RESUME_OVERRIDE_PATTERNS = (
    re.compile(r"checkpoint\.resume_from\b"),
    re.compile(r"resume(?:d|s|ing)?[\w\s:=-]{0,80}checkpoint", re.IGNORECASE),
    re.compile(r"checkpoint[\w\s:=-]{0,80}resume(?:d|s|ing)?", re.IGNORECASE),
)


def _code_blocks(text: str) -> list[str]:
    return [match.group(1) for match in re.finditer(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)]


class TestTrainingSurface:
    def test_tinker_integration_targets_are_absent_from_configs(self):
        for path in sorted((REPO_ROOT / "src" / "synthstats" / "configs").rglob("*.yaml")):
            text = path.read_text()
            assert "synthstats.integrations.tinker" not in text, path

    def test_docs_use_supported_runner_commands(self):
        for path in DOC_FILES:
            for block in _code_blocks(path.read_text()):
                for token in INVALID_RUNNER_TOKENS:
                    assert token not in block, f"{path} contains unsupported override {token}"

    def test_scripts_use_supported_runner_commands(self):
        for path in SCRIPT_FILES:
            text = path.read_text()
            for token in INVALID_RUNNER_TOKENS:
                assert token not in text, f"{path} contains unsupported override {token}"

    def test_runner_configs_use_task_runtime_env_id(self):
        for path in RUNNER_CONFIG_FILES:
            text = path.read_text()
            assert "env_class: synthstats-task-runtime-v0" in text, (
                f"{path} must configure env_class: synthstats-task-runtime-v0"
            )

    def test_env_configs_use_statistical_modeling_task_target(self):
        for path in ENV_CONFIG_FILES:
            text = path.read_text()
            assert (
                "_target_: synthstats.tasks.statistical_modeling.task.StatisticalModelingTask"
                in text
            ), f"{path} must target StatisticalModelingTask"

    def test_task_configs_use_statistical_modeling_task_name(self):
        for path in TASK_CONFIG_FILES:
            text = path.read_text()
            assert "name: statistical_modeling" in text, (
                f"{path} must set name: statistical_modeling"
            )

    def test_train_config_preserves_env_owned_judge_defaults(self):
        text = TRAIN_CONFIG_FILE.read_text()
        assert "- judge: hybrid_lm" not in text, (
            "configs/train.yaml must preserve env-owned judge defaults"
        )

    def test_top_level_docs_describe_statistical_modeling_tasks(self):
        for path in DOC_FILES:
            text = path.read_text()
            assert "StatisticalModelingTask" in text or "dugongs" in text, (
                f"{path} must document the statistical modeling tasks"
            )

    def test_training_entrypoints_omit_checkpoint_resume_examples(self):
        for path in [*DOC_FILES, *SCRIPT_FILES]:
            text = path.read_text()
            for pattern in RESUME_OVERRIDE_PATTERNS:
                assert not pattern.search(text), f"{path} matches {pattern.pattern!r}"
