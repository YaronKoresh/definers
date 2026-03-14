from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from definers.platform import runtime


def test_get_os_name_lowercases_platform_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime.platform, "system", lambda: "Windows")

    assert runtime.get_os_name() == "windows"


def test_is_admin_windows_returns_shell_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime.ctypes,
        "windll",
        SimpleNamespace(shell32=SimpleNamespace(IsUserAnAdmin=lambda: True)),
        raising=False,
    )

    assert runtime.is_admin_windows() is True


def test_is_admin_windows_returns_false_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime.ctypes, "windll", object(), raising=False)

    assert runtime.is_admin_windows() is False


def test_cores_returns_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 12)

    assert runtime.cores() == 12


def test_get_python_version_formats_version_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(version_info=SimpleNamespace(major=3, minor=12, micro=7)))

    assert runtime.get_python_version() == "3.12.7"


def test_get_python_version_returns_none_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "sys", object())

    assert runtime.get_python_version() is None


def test_importable_rejects_non_string_name() -> None:
    assert runtime.importable(1) is False                          


def test_importable_strips_name_before_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    looked_up: list[str] = []
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda name: looked_up.append(name) or object())

    assert runtime.importable("  json  ") is True
    assert looked_up == ["json"]


def test_importable_returns_false_when_find_spec_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_lookup_error(name: str) -> object:
        raise RuntimeError("bad import")

    monkeypatch.setattr(runtime.importlib.util, "find_spec", raise_lookup_error)

    assert runtime.importable("broken") is False


def test_runnable_rejects_blank_command() -> None:
    assert runtime.runnable("  ") is False


def test_runnable_uses_which_for_first_token(monkeypatch: pytest.MonkeyPatch) -> None:
    looked_up: list[str] = []
    monkeypatch.setattr(runtime.shutil, "which", lambda name: looked_up.append(name) or "/bin/python")

    assert runtime.runnable('"python" -V') is True
    assert looked_up == ["python"]


def test_runnable_falls_back_to_simple_split_when_shlex_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __import__

    class BrokenShlex:
        @staticmethod
        def split(command_line: str, posix: bool = False) -> list[str]:
            raise ValueError("bad syntax")

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "shlex":
            return BrokenShlex
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.setattr(runtime.shutil, "which", lambda name: "/usr/bin/cmd" if name == "cmd" else None)

    assert runtime.runnable("cmd /c dir") is True


def test_check_version_wildcard_handles_none_values() -> None:
    assert runtime.check_version_wildcard(None, None) is True
    assert runtime.check_version_wildcard("1.0", None) is False


def test_normalize_name_and_version_matches() -> None:
    assert runtime._normalize_name("  NumPy  ") == "numpy"
    assert runtime._normalize_name(None) == ""
    assert runtime._version_matches(None, None) is True
    assert runtime._version_matches("1.2", "1.2.3") is True
    assert runtime._version_matches("1.2.*", "1.2.3") is True
    assert runtime._version_matches("1.2", "") is False