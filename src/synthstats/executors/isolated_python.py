"""Linux namespace isolation for submitted Python programs."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from synthstats.core.process import cleanup_process_group, set_subprocess_rlimits


def isolated_python_command() -> list[str]:
    """Expose the interpreter and libraries read-only, with private temporary storage."""
    bwrap = shutil.which("bwrap")
    if sys.platform != "linux" or bwrap is None:
        raise RuntimeError(
            "Program execution requires Linux with bubblewrap (bwrap). "
            "Install bubblewrap in the execution environment."
        )
    command = [
        bwrap,
        "--unshare-all",
        "--unshare-user",
        "--disable-userns",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--clearenv",
    ]
    roots = {Path("/usr"), Path("/lib"), Path("/lib64"), Path(sys.prefix), Path(sys.base_prefix)}
    interpreter = Path(sys.executable)
    while interpreter.is_symlink():
        target = interpreter.readlink()
        if target.is_absolute():
            roots.add(target.parent.parent)
            interpreter = target
        else:
            interpreter = interpreter.parent / target
    for root in sorted(roots):
        if root.exists():
            command.extend(["--ro-bind", str(root), str(root)])
    command.extend(
        [
            "--ro-bind-try",
            "/etc/ld.so.cache",
            "/etc/ld.so.cache",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/dev/shm",
            "--tmpfs",
            "/tmp",
            "--chdir",
            "/tmp",
            "--remount-ro",
            "/",
        ]
    )
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTENSOR_FLAGS": "device=cpu,cxx=,base_compiledir=/tmp/pytensor",
        "MPLCONFIGDIR": "/tmp/matplotlib",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    for key, value in environment.items():
        command.extend(["--setenv", key, value])
    return [*command, sys.executable, "-"]


def run_isolated_python(code: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run one request in a fresh interpreter and drain both output pipes."""
    command = isolated_python_command()
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        preexec_fn=lambda: set_subprocess_rlimits(timeout),
    )
    try:
        stdout, stderr = proc.communicate(input=code, timeout=timeout)
        return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)
    finally:
        if proc.poll() is None:
            cleanup_process_group(proc)
