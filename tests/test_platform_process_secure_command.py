from __future__ import annotations

import os

import pytest

import definers.system.processes as processes


def normalize_absolute_path(path: str) -> str:
    return os.path.normcase(
        os.path.normpath(os.path.abspath(os.path.expanduser(path)))
    )


def test_secure_command_allows_current_python_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python_path = os.path.normpath(
        "C:/Users/User/AppData/Local/Programs/Python/Python310/python.exe"
    )
    monkeypatch.setattr(processes.sys, "executable", python_path)

    result = processes.secure_command([python_path, "-m", "pip", "--version"])

    assert result == [
        normalize_absolute_path(python_path),
        "-m",
        "pip",
        "--version",
    ]


def test_secure_command_allows_absolute_executable_resolved_from_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cmake_path = os.path.normpath("/usr/bin/cmake")
    monkeypatch.setattr(processes.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(
        processes.shutil,
        "which",
        lambda name: cmake_path if name == "cmake" else None,
    )

    result = processes.secure_command([cmake_path, "--version"])

    assert result == [normalize_absolute_path(cmake_path), "--version"]


def test_secure_command_still_uses_secure_path_for_untrusted_absolute_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_secure_path(path: str) -> str:
        calls.append(path)
        return "C:/safe/tool.exe"

    monkeypatch.setattr(processes.sys, "executable", "C:/Python/python.exe")
    monkeypatch.setattr(processes.shutil, "which", lambda name: None)
    monkeypatch.setattr(processes, "secure_path", fake_secure_path)

    result = processes.secure_command(["C:/outside/tool.exe", "--help"])

    assert result == ["C:/safe/tool.exe", "--help"]
    assert calls == ["C:/outside/tool.exe"]


def test_secure_command_splits_safe_string() -> None:
    assert processes.secure_command("python -V") == ["python", "-V"]


def test_secure_command_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="Command is empty"):
        processes.secure_command("   ")


@pytest.mark.parametrize(
    "command", ["echo ok; whoami", "echo ok | whoami", "echo `x`", "echo $HOME"]
)
def test_secure_command_rejects_unsafe_characters(command: str) -> None:
    with pytest.raises(ValueError, match="Unsafe characters"):
        processes.secure_command(command)


def test_secure_command_rejects_invalid_syntax() -> None:
    with pytest.raises(ValueError, match="Invalid command syntax"):
        processes.secure_command('echo "unterminated')


def test_secure_command_normalizes_sequence() -> None:
    assert processes.secure_command(["  python  ", "", "  -m  ", " pip "]) == [
        "python",
        "-m",
        "pip",
    ]


def test_secure_command_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="Command list is empty"):
        processes.secure_command([" ", ""])


def test_secure_command_rejects_overlong_argument() -> None:
    with pytest.raises(ValueError, match="Argument too long"):
        processes.secure_command(["python", "x" * 1025])


def test_secure_command_uses_secure_path_for_pathlike_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_secure_path(path: str) -> str:
        calls.append(path)
        return "C:/safe/tool.exe"

    monkeypatch.setattr(processes, "secure_path", fake_secure_path)

    result = processes.secure_command(["  ./tool.exe  ", "--help"])

    assert result == ["C:/safe/tool.exe", "--help"]
    assert calls == ["./tool.exe"]


def test_secure_command_rejects_invalid_input_type() -> None:
    with pytest.raises(TypeError, match="Command must be a string or a list"):
        processes.secure_command(42)
