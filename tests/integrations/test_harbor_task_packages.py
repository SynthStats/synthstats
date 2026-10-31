import re
import shlex
import tomllib
from pathlib import Path

TASK_PACKAGE_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "synthstats"
    / "integrations"
    / "harbor"
    / "task_packages"
)
UV_LOCK_PATH = Path(__file__).resolve().parents[2] / "uv.lock"
EXPECTED_BASE_IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:"
    "e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58"
)
REQUIREMENT_NAMES = (
    "numpy",
    "scipy",
    "pandas",
    "hydra-core",
    "omegaconf",
    "pymc",
    "arviz",
)
REQUIREMENT_SPEC_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)(?P<operator>===|==|~=|!=|<=|>=|<|>)(?P<version>.+)$"
)


def _locked_package_versions() -> dict[str, str]:
    lock_text = UV_LOCK_PATH.read_text(encoding="utf-8")
    versions: dict[str, str] = {}

    for package_name in REQUIREMENT_NAMES:
        match = re.search(
            rf'^name = "{re.escape(package_name)}"\nversion = "([^"]+)"',
            lock_text,
            flags=re.MULTILINE,
        )
        assert match is not None, f"expected {package_name} to be present in uv.lock"
        versions[package_name] = match.group(1)

    return versions


def _pip_install_requirements(contents: str) -> list[tuple[str, str, str]]:
    normalized = contents.replace("\\\n", " ")
    for line in normalized.splitlines():
        stripped = line.strip()
        if not stripped.startswith("RUN uv pip install --system --no-cache "):
            continue
        tokens = shlex.split(stripped.removeprefix("RUN "))
        requirement_specs: list[tuple[str, str, str]] = []
        for token in tokens[3:]:
            match = REQUIREMENT_SPEC_RE.match(token)
            if match is None:
                continue
            requirement_specs.append(
                (match.group("name"), match.group("operator"), match.group("version"))
            )
        return requirement_specs
    raise AssertionError("expected Dockerfile to contain a pip install command")


def test_task_package_dockerfiles_pin_runtime_dependencies() -> None:
    dockerfiles = sorted(TASK_PACKAGE_DIR.glob("*/environment/Dockerfile"))
    expected_requirement_pins = _locked_package_versions()

    assert dockerfiles, "expected Harbor task-package Dockerfiles"

    for dockerfile in dockerfiles:
        contents = dockerfile.read_text(encoding="utf-8")
        first_line = contents.splitlines()[0].strip()
        assert first_line == f"FROM {EXPECTED_BASE_IMAGE}", (
            f"{dockerfile} must pin the Harbor Python base image exactly"
        )
        assert "bubblewrap=0.8.0-2+deb12u1" in contents
        assert "ca-certificates=20250419~deb12u2" in contents

        quoted_requirements = _pip_install_requirements(contents)
        for package_name, version in expected_requirement_pins.items():
            package_specs = [spec for spec in quoted_requirements if spec[0] == package_name]
            assert package_specs == [(package_name, "==", version)], (
                f"{dockerfile} must pin {package_name} exactly once to uv.lock version {version}"
            )


def test_task_package_compose_overrides_disable_registry_pulls() -> None:
    compose_files = sorted(TASK_PACKAGE_DIR.glob("*/environment/docker-compose.yaml"))

    assert compose_files, "expected Harbor task-package compose overrides"

    for compose_file in compose_files:
        contents = compose_file.read_text(encoding="utf-8")
        assert "pull_policy: never" in contents, (
            f"{compose_file} must disable registry pulls for local Harbor task images"
        )


def test_task_packages_declare_stable_local_image_tags() -> None:
    task_tomls = sorted(TASK_PACKAGE_DIR.glob("*/task.toml"))

    assert task_tomls, "expected Harbor task-package manifests"

    for task_toml in task_tomls:
        task_name = task_toml.parent.name
        payload = tomllib.loads(task_toml.read_text(encoding="utf-8"))
        image_name = payload["environment"]["docker_image"]
        assert image_name == f"synthstats-harbor-e2e:{task_name}"
