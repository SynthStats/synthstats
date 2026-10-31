import os
import signal
from unittest.mock import MagicMock, patch

from synthstats.core.process import (
    _group_exited,
    _wait_group_exit,
    cleanup_process_group,
)


def test_group_exited_missing_pid():
    assert _group_exited(2_000_000_000) is True


def test_group_exited_self_alive():
    assert _group_exited(os.getpgrp()) is False


def test_wait_group_exit_already_gone():
    assert _wait_group_exit(2_000_000_000, timeout_s=0.05) is True


def test_cleanup_sends_sigterm_then_stops_if_exited():
    proc = MagicMock()
    proc.pid = 12345

    with (
        patch("synthstats.core.process.os.getpgid", return_value=99),
        patch("synthstats.core.process.os.killpg") as mock_killpg,
        patch("synthstats.core.process._wait_group_exit", return_value=True),
    ):
        cleanup_process_group(proc)

    mock_killpg.assert_called_once_with(99, signal.SIGTERM)
    proc.kill.assert_called_once()
    proc.wait.assert_called_once()


def test_cleanup_escalates_to_sigkill():
    proc = MagicMock()
    proc.pid = 12345

    with (
        patch("synthstats.core.process.os.getpgid", return_value=99),
        patch("synthstats.core.process.os.killpg") as mock_killpg,
        patch("synthstats.core.process._wait_group_exit", side_effect=[False, True]),
    ):
        cleanup_process_group(proc)

    assert mock_killpg.call_count == 2
    mock_killpg.assert_any_call(99, signal.SIGTERM)
    mock_killpg.assert_any_call(99, signal.SIGKILL)


def test_cleanup_fallback_when_pgid_fails():
    proc = MagicMock()
    proc.pid = 12345

    with patch("synthstats.core.process.os.getpgid", side_effect=ProcessLookupError("gone")):
        cleanup_process_group(proc)

    proc.kill.assert_called_once()
    proc.wait.assert_called_once()
