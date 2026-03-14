from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from definers.platform import runtime


def test_installed_rejects_blank_package_name() -> None:
    assert runtime.installed("   ") is False


def test_installed_returns_true_from_windows_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    run_calls: list[object] = []
    monkeypatch.setattr(runtime, "get_os_name", lambda: "windows")
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda command, **kwargs: run_calls.append(command) or SimpleNamespace(stdout="Example Tool  1.2.3\n"),
    )

    assert runtime.installed("example tool", "1.2") is True
    assert len(run_calls) == 1


def test_installed_falls_back_to_pip_after_windows_registry_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(runtime, "get_os_name", lambda: "windows")

    def fake_run(command: object, **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        if isinstance(command, str):
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout="Package    Version\nexample-tool  4.5.6\n")

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)

    assert runtime.installed("example-tool", "4.5") is True
    assert len(calls) == 2


def test_installed_returns_true_for_linux_binary_version(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(runtime, "get_os_name", lambda: "linux")
    monkeypatch.setattr(runtime.shutil, "which", lambda name: "/usr/bin/tool" if name == "tool" else None)

    def fake_run(command: object, **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout="tool version 2.4.1\n")

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)

    assert runtime.installed("tool", "2.4") is True
    assert calls == [["tool", "--version"]]


def test_installed_uses_linux_v_fallback_when_version_output_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(runtime, "get_os_name", lambda: "linux")
    monkeypatch.setattr(runtime.shutil, "which", lambda name: "/usr/bin/tool" if name == "tool" else None)

    def fake_run(command: object, **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        if command == ["tool", "--version"]:
            return SimpleNamespace(stdout="")
        return SimpleNamespace(stdout="tool 3.1.0\n")

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)

    assert runtime.installed("tool", "3.1") is True
    assert calls == [["tool", "--version"], ["tool", "-v"]]


def test_installed_returns_true_from_pip_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "get_os_name", lambda: "darwin")
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(stdout="Package    Version\nrequests  2.31.0\n"),
    )

    assert runtime.installed("requests", "2.31.*") is True


def test_installed_returns_false_when_nothing_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "get_os_name", lambda: "linux")
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(stdout="Package    Version\nother  1.0.0\n"),
    )

    assert runtime.installed("missing", "1.0") is False