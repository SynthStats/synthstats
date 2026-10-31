"""Architecture guardrails enforced via static import inspection."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "synthstats"


def _package_files(package_name: str) -> list[tuple[str, Path]]:
    package_root = PACKAGE_ROOT / package_name
    modules: list[tuple[str, Path]] = []

    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT).with_suffix("")
        parts = relative.parts
        if parts[-1] == "__init__":
            module_name = "synthstats." + ".".join(parts[:-1])
        else:
            module_name = "synthstats." + ".".join(parts)
        modules.append((module_name, path))

    return modules


def _module_package(module_name: str, path: Path) -> str:
    if path.name == "__init__.py":
        return module_name
    return module_name.rsplit(".", 1)[0]


def _resolve_relative_base(package_name: str, level: int) -> str:
    parts = package_name.split(".")
    return ".".join(parts[: len(parts) - level + 1])


def _import_targets(module_name: str, path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    current_package = _module_package(module_name, path)
    targets: list[tuple[int, str]] = []
    module_aliases: dict[str, str] = {}
    function_aliases: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_aliases[alias.asname or alias.name] = alias.name
                targets.append((node.lineno, alias.name))
            continue

        if not isinstance(node, ast.ImportFrom):
            continue

        if node.level == 0:
            base = node.module or ""
        else:
            relative_base = _resolve_relative_base(current_package, node.level)
            if node.module:
                base = f"{relative_base}.{node.module}"
            else:
                base = relative_base

        if base:
            targets.append((node.lineno, base))

        for alias in node.names:
            if alias.name == "*":
                continue
            full_name = f"{base}.{alias.name}" if base else alias.name
            function_aliases[alias.asname or alias.name] = full_name
            if base:
                targets.append((node.lineno, f"{base}.{alias.name}"))
            else:
                targets.append((node.lineno, alias.name))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue

        target: str | None = None
        if isinstance(node.func, ast.Name):
            if node.func.id == "__import__":
                target = "__import__"
            else:
                aliased = function_aliases.get(node.func.id)
                if aliased == "importlib.import_module":
                    target = aliased
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            module_name_for_alias = module_aliases.get(node.func.value.id, node.func.value.id)
            if module_name_for_alias == "importlib" and node.func.attr == "import_module":
                target = "importlib.import_module"

        if target is None:
            continue

        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            targets.append((node.lineno, first_arg.value))

    return targets


def _find_forbidden_imports(package_name: str, forbidden_prefix: str) -> list[str]:
    violations: list[str] = []

    for module_name, path in _package_files(package_name):
        for lineno, target in _import_targets(module_name, path):
            if target == forbidden_prefix or target.startswith(f"{forbidden_prefix}."):
                violations.append(f"{module_name}:{lineno} imports {target}")

    return violations


def _format_failure_message(rule: str, violations: list[str]) -> str:
    return f"{rule}\nViolations:\n" + "\n".join(violations)


def test_train_modules_do_not_import_tasks() -> None:
    violations = _find_forbidden_imports("train", "synthstats.tasks")
    assert not violations, _format_failure_message(
        "`synthstats.train.*` must not import `synthstats.tasks.*`",
        violations,
    )


def test_train_modules_do_not_import_integrations() -> None:
    violations = _find_forbidden_imports("train", "synthstats.integrations")
    assert not violations, _format_failure_message(
        "`synthstats.train.*` must not import `synthstats.integrations.*`",
        violations,
    )


def test_task_modules_do_not_import_train() -> None:
    violations = _find_forbidden_imports("tasks", "synthstats.train")
    assert not violations, _format_failure_message(
        "`synthstats.tasks.*` must not import `synthstats.train.*`",
        violations,
    )


def test_executor_modules_do_not_import_tasks() -> None:
    violations = _find_forbidden_imports("executors", "synthstats.tasks")
    assert not violations, _format_failure_message(
        "`synthstats.executors.*` should not depend on task modules",
        violations,
    )
