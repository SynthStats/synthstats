"""PyMC sandbox executor."""

from __future__ import annotations

import ast
import logging
import subprocess

from synthstats.core.executor import ToolResult
from synthstats.core.types import ToolCall
from synthstats.executors.isolated_python import run_isolated_python

logger = logging.getLogger(__name__)

# modules that could enable file system, network, or shell access
BLOCKED_MODULES = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "shutil",
        "pathlib",
        "ctypes",
        "multiprocessing",
        "threading",
        "signal",
        "pty",
        "tty",
        "termios",
        "fcntl",
        "resource",
        "syslog",
        "platform",
        "webbrowser",
        "http",
        "urllib",
        "ftplib",
        "smtplib",
        "poplib",
        "imaplib",
        "nntplib",
        "telnetlib",
        "xmlrpc",
        "asyncio",
        "concurrent",
        # dynamic import modules (gadget chain enablers)
        "importlib",
        "pkgutil",
        "types",
        "inspect",
        # I/O modules that bypass open()
        "io",
        "builtins",
        "code",
        "codeop",
        "runpy",
        # network and file bypass modules
        "requests",
        "tempfile",
        # memory introspection
        "gc",
        # persistence via exit hooks
        "atexit",
        # memory mapping bypass
        "mmap",
        # file creation bypasses
        "sqlite3",
        "logging",
        # crypto (reduces attack surface)
        "ssl",
        # serialization (RCE via __reduce__, defense in depth)
        "pickle",
        "_pickle",
    }
)

# built-in functions that allow code execution or file access
BLOCKED_FUNCTIONS = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "type",
        "input",
        "breakpoint",
    }
)

# dangerous methods on allowed modules (pandas, numpy, scipy, torch)
# that bypass open() by implementing their own file I/O
BLOCKED_METHODS = frozenset(
    {
        # pandas file I/O - read
        "read_csv",
        "read_pickle",
        "read_json",
        "read_excel",
        "read_parquet",
        "read_html",
        "read_xml",
        "read_sql",
        "read_table",
        "read_fwf",
        "read_stata",
        "read_sas",
        "read_spss",
        "read_feather",
        "read_orc",
        # pandas file I/O - write
        "to_csv",
        "to_pickle",
        "to_json",
        "to_excel",
        "to_parquet",
        "to_html",
        "to_sql",
        "to_stata",
        "to_feather",
        # numpy file I/O
        "load",
        "loadtxt",
        "genfromtxt",
        "fromfile",
        "save",
        "savetxt",
        "savez",
        "savez_compressed",
        "tofile",
        # scipy file I/O
        "loadmat",
        "savemat",
        "wavread",
        "wavwrite",
        # torch file I/O (pickle-based, RCE risk)
        "load_state_dict",
        "save_state_dict",
        # ctypes/shared library loading
        "load_library",
        "CDLL",
        "WinDLL",
        "PyDLL",
        "OleDLL",
        # pickle operations (RCE)
        "loads",
        "dumps",
        "dump",
    }
)


def rewrite_pm_sample_calls(source: str, *, random_seed: int | None = None) -> str:
    """Rewrite pm.sample() calls to bounded settings using AST (not regex).

    Set draws=200, chains=2 and cores=1 for rollout scoring.
    Strict PSIS-LOO diagnostics screen the resulting fits.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    class _SampleCallRewriter(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "pm"
                and node.func.attr == "sample"
            ):
                args = list(node.args)
                keywords = [
                    kw for kw in node.keywords if kw.arg not in {"draws", "chains", "cores"}
                ]

                if args:
                    args[0] = ast.Constant(value=200)
                else:
                    keywords.append(ast.keyword(arg="draws", value=ast.Constant(value=200)))

                keywords.append(ast.keyword(arg="chains", value=ast.Constant(value=2)))
                keywords.append(ast.keyword(arg="cores", value=ast.Constant(value=1)))
                if random_seed is not None:
                    keywords = [kw for kw in keywords if kw.arg != "random_seed"]
                    keywords.append(
                        ast.keyword(arg="random_seed", value=ast.Constant(value=random_seed))
                    )
                node.args = args
                node.keywords = keywords
            return node

    try:
        rewritten = _SampleCallRewriter().visit(tree)
        ast.fix_missing_locations(rewritten)
        return ast.unparse(rewritten)
    except (TypeError, ValueError, RecursionError) as exc:
        logger.debug("AST rewrite failed, returning original code: %s", exc)
        return source


class ASTSecurityChecker(ast.NodeVisitor):
    """Visitor that detects dangerous code patterns.

    Uses defence in depth: blocks known dangerous patterns AND uses allowlist
    for attribute access to prevent gadget chain attacks.
    """

    # allowlisted module prefixes for attribute access
    ALLOWED_MODULES = frozenset(
        {
            "pm",
            "pymc",
            "pytensor",
            "np",
            "numpy",
            "math",
            "scipy",
            "arviz",
            "az",
            "xarray",
            "pandas",
            "pd",
            "jax",
            "numpyro",
            "torch",
            "time",
        }
    )

    def __init__(self):
        self.violations: list[str] = []
        # map imported alias -> root module for allowlist checks
        self._import_aliases: dict[str, str] = {}
        self._imported_symbols: dict[str, str] = {}

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module_name = alias.name.split(".")[0]
            if module_name not in self.ALLOWED_MODULES or module_name in BLOCKED_MODULES:
                self.violations.append(f"Blocked import: {alias.name}")
            else:
                alias_name = alias.asname or module_name
                self._import_aliases[alias_name] = module_name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or not node.module:
            self.violations.append("Blocked relative import")
            return
        if node.module:
            module_name = node.module.split(".")[0]
            if module_name not in self.ALLOWED_MODULES or module_name in BLOCKED_MODULES:
                self.violations.append(f"Blocked import: from {node.module}")
            else:
                for alias in node.names:
                    if alias.name == "*" or alias.name.startswith("__"):
                        self.violations.append(f"Blocked import: {alias.name}")
                    alias_name = alias.asname or alias.name
                    self._import_aliases[alias_name] = module_name
                    self._imported_symbols[alias_name] = alias.name
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # check for direct function calls like open(), exec(), loadmat()
        # also catches `from scipy.io import loadmat; loadmat('file')`
        if isinstance(node.func, ast.Name):
            symbol = self._imported_symbols.get(node.func.id, node.func.id)
            if symbol in BLOCKED_FUNCTIONS:
                self.violations.append(f"Blocked function call: {node.func.id}()")
            elif symbol in BLOCKED_METHODS:
                self.violations.append(
                    f"Blocked file I/O function: {node.func.id}() - use PyMC for modeling only"
                )
        # check for method calls - block both BLOCKED_FUNCTIONS and BLOCKED_METHODS
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in BLOCKED_FUNCTIONS:
                self.violations.append(f"Blocked function call: {node.func.attr}()")
            elif node.func.attr in BLOCKED_METHODS:
                self.violations.append(
                    f"Blocked file I/O method: {node.func.attr}() - use PyMC for modeling only"
                )
        # CRITICAL: block calls on complex expressions (gadget chains)
        # e.g., ().__class__.__base__.__subclasses__()[0]()
        elif isinstance(node.func, ast.Subscript):
            self.violations.append("Blocked: calls on subscript expressions are not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Reject every attribute write, including aliases and container lookups:
        # syntactic import-name tracking cannot prove an object is not a module.
        if isinstance(node.ctx, ast.Store | ast.Del):
            self.violations.append(f"Blocked attribute mutation: {node.attr}")

        # block dunder attribute access (used in gadget chains)
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.violations.append(f"Blocked dunder access: {node.attr}")

        # enforce allowlist for imported module aliases (e.g., pd.read_csv)
        root = self._resolve_root_name(node)
        if root is not None and root in self._import_aliases:
            module_name = self._import_aliases[root]
            if module_name not in self.ALLOWED_MODULES:
                self.violations.append(
                    "Blocked attribute access via non-allowlisted module alias: "
                    f"{root} ({module_name})"
                )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        # block loading of dangerous function names (prevents aliasing bypass)
        # e.g., `imp = __import__; imp("os")` - blocks loading __import__ into imp
        if isinstance(node.ctx, ast.Load):
            if node.id in BLOCKED_FUNCTIONS or node.id.startswith("__"):
                self.violations.append(
                    f"Blocked: cannot reference '{node.id}' (prevents aliasing bypass)"
                )
        self.generic_visit(node)

    @staticmethod
    def _resolve_root_name(node: ast.AST) -> str | None:
        cur = node
        while isinstance(cur, ast.Attribute):
            cur = cur.value
        if isinstance(cur, ast.Name):
            return cur.id
        return None


def check_code_safety(code: str) -> tuple[bool, str | None]:
    """Returns (is_safe, error_message). error_message is None when safe."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    checker = ASTSecurityChecker()
    checker.visit(tree)

    if checker.violations:
        return False, f"Blocked: {'; '.join(checker.violations)}"

    return True, None


class PyMCExecutor:
    """Runs PyMC programs in a sandboxed subprocess with AST safety checks."""

    name: str = "pymc"

    def execute(self, payload: ToolCall, *, timeout_s: float) -> ToolResult:
        code = payload.input.get("code")
        if code is None:
            return ToolResult(
                output="",
                success=False,
                error="Missing 'code' key in input",
            )

        is_safe, error = check_code_safety(code)
        if not is_safe:
            return ToolResult(
                output="",
                success=False,
                error=error,
            )

        return self._run_subprocess(code, timeout_s)

    def _run_subprocess(self, code: str, timeout_s: float) -> ToolResult:
        try:
            result = run_isolated_python(code, timeout=timeout_s)
            return ToolResult(
                output=result.stdout,
                success=result.returncode == 0,
                error=(result.stderr.strip() or "Program failed") if result.returncode else None,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                output="", success=False, error=f"Timeout: code execution exceeded {timeout_s}s"
            )
        except (RuntimeError, subprocess.SubprocessError, OSError) as e:
            return ToolResult(
                output="",
                success=False,
                error=f"Execution setup failed: {e}",
            )
