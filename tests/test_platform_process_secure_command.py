from __future__ import annotations

import pytest

from definers.platform import processes


def test_secure_command_splits_safe_string() -> None:
    assert processes.secure_command("python -V") == ["python", "-V"]


def test_secure_command_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="Command is empty"):
        processes.secure_command("   ")


@pytest.mark.parametrize("command", ["echo ok; whoami", "echo ok | whoami", "echo `x`", "echo $HOME"])
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


def test_secure_command_uses_secure_path_for_pathlike_executable(monkeypatch: pytest.MonkeyPatch) -> None:
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