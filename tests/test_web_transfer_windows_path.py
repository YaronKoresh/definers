import os
import sys
from types import ModuleType
from unittest.mock import MagicMock

from definers.media.web_transfer import add_to_path_windows


def build_winreg_module(
    current_path: str | BaseException | None = None,
    open_key_error: BaseException | None = None,
) -> ModuleType:
    module = ModuleType("winreg")
    module.HKEY_CURRENT_USER = 1
    module.KEY_ALL_ACCESS = 2
    module.REG_EXPAND_SZ = 3
    module.OpenKey = MagicMock()
    module.QueryValueEx = MagicMock()
    module.SetValueEx = MagicMock()
    module.CloseKey = MagicMock()
    if open_key_error is not None:
        module.OpenKey.side_effect = open_key_error
    else:
        module.OpenKey.return_value = object()
    if isinstance(current_path, BaseException):
        module.QueryValueEx.side_effect = current_path
    elif current_path is None:
        module.QueryValueEx.return_value = ("", module.REG_EXPAND_SZ)
    else:
        module.QueryValueEx.return_value = (current_path, module.REG_EXPAND_SZ)
    return module


def test_add_to_path_windows_noops_outside_windows(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("PATH", "BASE")

    add_to_path_windows("C:\\Tools")

    assert os.environ["PATH"] == "BASE"


def test_add_to_path_windows_adds_missing_folder_and_broadcasts(
    monkeypatch,
) -> None:
    broadcaster_calls: list[str] = []
    winreg_module = build_winreg_module(current_path=FileNotFoundError())
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "winreg", winreg_module)
    monkeypatch.setenv("PATH", "Existing")

    add_to_path_windows(
        '"C:\\Tools\\Bin"',
        broadcaster=lambda: broadcaster_calls.append("broadcasted"),
    )

    assert winreg_module.SetValueEx.call_count == 1
    assert winreg_module.CloseKey.call_count == 1
    assert broadcaster_calls == ["broadcasted"]
    assert os.environ["PATH"].startswith("C:\\Tools\\Bin" + os.pathsep)
    assert winreg_module.SetValueEx.call_args[0][4] == "C:\\Tools\\Bin"


def test_add_to_path_windows_skips_duplicate_folder(monkeypatch) -> None:
    broadcaster_calls: list[str] = []
    existing_folder = os.path.normpath("C:\\Tools\\Bin")
    winreg_module = build_winreg_module(
        current_path=existing_folder + ";C:\\Other\\Bin"
    )
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "winreg", winreg_module)
    monkeypatch.setenv("PATH", existing_folder + os.pathsep + "C:\\Other\\Bin")

    add_to_path_windows(
        existing_folder,
        broadcaster=lambda: broadcaster_calls.append("broadcasted"),
    )

    assert winreg_module.SetValueEx.call_count == 0
    assert broadcaster_calls == []


def test_add_to_path_windows_suppresses_registry_errors(monkeypatch) -> None:
    winreg_module = build_winreg_module(
        open_key_error=PermissionError("denied")
    )
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "winreg", winreg_module)

    add_to_path_windows("C:\\Tools\\Bin")
