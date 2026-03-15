from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from definers.platform import processes


def test_run_command_returns_stdout_lines_and_merges_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged: list[tuple[str, str]] = []
    monkeypatch.setitem(
        sys.modules,
        "definers.file_ops",
        SimpleNamespace(
            log=lambda message, detail: logged.append((message, detail)),
            catch=lambda error: None,
        ),
    )

    process = MagicMock()
    process.communicate.return_value = (" first\n\nsecond \n", "")
    process.returncode = 0

    popen_calls: list[dict[str, object]] = []

    def fake_popen(*args: object, **kwargs: object) -> MagicMock:
        popen_calls.append({"args": args, "kwargs": kwargs})
        return process

    monkeypatch.setattr(processes.subprocess, "Popen", fake_popen)

    result = processes._run_command(
        ["python", "-V"], silent=True, env={"LANE": "4"}
    )

    assert result == ["first", "second"]
    assert popen_calls[0]["args"] == (["python", "-V"],)
    assert popen_calls[0]["kwargs"]["env"] == {**os.environ, "LANE": "4"}
    assert logged == []


def test_run_command_rejects_unsafe_command_and_reports_catch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caught: list[Exception] = []
    monkeypatch.setitem(
        sys.modules,
        "definers.file_ops",
        SimpleNamespace(
            log=lambda message, detail: None,
            catch=lambda error: caught.append(error),
        ),
    )
    popen = MagicMock()
    monkeypatch.setattr(processes.subprocess, "Popen", popen)

    result = processes._run_command("echo ok; whoami", silent=True)

    assert result is False
    assert len(caught) == 1
    assert isinstance(caught[0], ValueError)
    popen.assert_not_called()


def test_run_command_returns_false_and_logs_on_non_zero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged: list[tuple[str, str]] = []
    monkeypatch.setitem(
        sys.modules,
        "definers.file_ops",
        SimpleNamespace(
            log=lambda message, detail: logged.append((message, detail)),
            catch=lambda error: None,
        ),
    )

    process = MagicMock()
    process.communicate.return_value = ("out\n", "bad\n")
    process.returncode = 2
    monkeypatch.setattr(
        processes.subprocess, "Popen", lambda *args, **kwargs: process
    )

    result = processes._run_command(["python", "bad.py"], silent=False)

    assert result is False
    assert logged == [
        ("Script failed [2]", "python bad.py"),
        ("Stderr: bad", ""),
    ]


def test_run_command_catches_popen_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caught: list[Exception] = []
    monkeypatch.setitem(
        sys.modules,
        "definers.file_ops",
        SimpleNamespace(
            log=lambda message, detail: None,
            catch=lambda error: caught.append(error),
        ),
    )

    def raise_runtime_error(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(processes.subprocess, "Popen", raise_runtime_error)

    result = processes._run_command(["python"], silent=True)

    assert result is False
    assert len(caught) == 1
    assert str(caught[0]) == "boom"


def test_run_linux_delegates_to_run_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, bool, object]] = []
    monkeypatch.setattr(
        processes,
        "_run_command",
        lambda command, silent=False, env=None: (
            calls.append((command, silent, env)) or ["ok"]
        ),
    )

    assert processes.run_linux("ls", silent=True, env={"A": "1"}) == ["ok"]
    assert calls == [("ls", True, {"A": "1"})]


def test_run_windows_delegates_to_run_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, bool, object]] = []
    monkeypatch.setattr(
        processes,
        "_run_command",
        lambda command, silent=False, env=None: (
            calls.append((command, silent, env)) or ["ok"]
        ),
    )

    assert processes.run_windows("dir", silent=False, env=None) == ["ok"]
    assert calls == [("dir", False, None)]


def test_run_uses_windows_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    run_windows = MagicMock(return_value=["win"])
    monkeypatch.setattr(processes, "run_windows", run_windows)

    result = processes.run("dir", silent=True, env={"X": "1"})

    assert result == ["win"]
    run_windows.assert_called_once_with("dir", silent=True, env={"X": "1"})


def test_run_uses_linux_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    run_linux = MagicMock(return_value=["linux"])
    monkeypatch.setattr(processes, "run_linux", run_linux)

    result = processes.run("ls", silent=False, env=None)

    assert result == ["linux"]
    run_linux.assert_called_once_with("ls", silent=False, env=None)


def test_get_process_pid_returns_parsed_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        processes.subprocess, "check_output", lambda args: b"12345\n"
    )

    assert processes.get_process_pid("python") == 12345


def test_get_process_pid_returns_none_for_missing_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_called_process_error(args: object) -> bytes:
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(
        processes.subprocess, "check_output", raise_called_process_error
    )

    assert processes.get_process_pid("missing") is None


def test_send_signal_to_process_returns_true_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        processes.os,
        "kill",
        lambda pid, signal_number: kill_calls.append((pid, signal_number)),
    )

    assert processes.send_signal_to_process(100, 15) is True
    assert kill_calls == [(100, 15)]


def test_send_signal_to_process_returns_false_on_os_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_os_error(pid: int, signal_number: int) -> None:
        raise OSError("denied")

    monkeypatch.setattr(processes.os, "kill", raise_os_error)

    assert processes.send_signal_to_process(100, 15) is False
