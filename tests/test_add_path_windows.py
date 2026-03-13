import os
import sys
import unittest
from unittest.mock import MagicMock, call, patch

from definers.web import add_to_path_windows


class TestAddToPathWindows(unittest.TestCase):
    @patch.object(sys, "platform", "linux")
    def test_noop_on_non_windows(self) -> None:
        original_path = os.environ.get("PATH", "")
        add_to_path_windows("C:\\some\\folder")
        self.assertEqual(os.environ.get("PATH", ""), original_path)

    @patch.object(sys, "platform", "darwin")
    def test_noop_on_macos(self) -> None:
        original_path = os.environ.get("PATH", "")
        add_to_path_windows("C:\\some\\folder")
        self.assertEqual(os.environ.get("PATH", ""), original_path)

    @patch.object(sys, "platform", "win32")
    @patch("definers.web.broadcast_path_change")
    def test_adds_new_folder_to_path(self, mock_broadcast: MagicMock) -> None:
        mock_key = MagicMock()
        existing_path = "C:\\Existing\\Bin;C:\\Another\\Bin"
        folder = "C:\\New\\Folder"

        winreg_mock = MagicMock()
        winreg_mock.HKEY_CURRENT_USER = 0x80000001
        winreg_mock.KEY_ALL_ACCESS = 0xF003F
        winreg_mock.REG_EXPAND_SZ = 2
        winreg_mock.OpenKey.return_value = mock_key
        winreg_mock.QueryValueEx.return_value = (existing_path, 2)

        with patch.dict("sys.modules", {"winreg": winreg_mock}):
            with patch.dict(os.environ, {"PATH": existing_path}):
                add_to_path_windows(folder)

                winreg_mock.OpenKey.assert_called_once()
                winreg_mock.SetValueEx.assert_called_once()
                set_call_args = winreg_mock.SetValueEx.call_args
                new_path_value = set_call_args[0][4]
                self.assertTrue(new_path_value.startswith(folder))
                mock_broadcast.assert_called_once()
                winreg_mock.CloseKey.assert_called_once_with(mock_key)

    @patch.object(sys, "platform", "win32")
    @patch("definers.web.broadcast_path_change")
    def test_skips_duplicate_folder(self, mock_broadcast: MagicMock) -> None:
        folder = os.path.normpath("C:\\Existing\\Bin")
        existing_path = f"{folder};C:\\Another\\Bin"

        winreg_mock = MagicMock()
        winreg_mock.HKEY_CURRENT_USER = 0x80000001
        winreg_mock.KEY_ALL_ACCESS = 0xF003F
        winreg_mock.OpenKey.return_value = MagicMock()
        winreg_mock.QueryValueEx.return_value = (existing_path, 2)

        with patch.dict("sys.modules", {"winreg": winreg_mock}):
            with patch.dict(os.environ, {"PATH": existing_path}):
                add_to_path_windows(folder)

                winreg_mock.SetValueEx.assert_not_called()
                mock_broadcast.assert_not_called()

    @patch.object(sys, "platform", "win32")
    @patch("definers.web.broadcast_path_change")
    def test_handles_empty_existing_path(
        self, mock_broadcast: MagicMock
    ) -> None:
        mock_key = MagicMock()
        folder = "C:\\New\\Folder"

        winreg_mock = MagicMock()
        winreg_mock.HKEY_CURRENT_USER = 0x80000001
        winreg_mock.KEY_ALL_ACCESS = 0xF003F
        winreg_mock.REG_EXPAND_SZ = 2
        winreg_mock.OpenKey.return_value = mock_key
        winreg_mock.QueryValueEx.side_effect = FileNotFoundError

        with patch.dict("sys.modules", {"winreg": winreg_mock}):
            with patch.dict(os.environ, {"PATH": ""}):
                add_to_path_windows(folder)

                winreg_mock.SetValueEx.assert_called_once()
                mock_broadcast.assert_called_once()

    @patch.object(sys, "platform", "win32")
    def test_handles_registry_error_gracefully(self) -> None:
        winreg_mock = MagicMock()
        winreg_mock.HKEY_CURRENT_USER = 0x80000001
        winreg_mock.KEY_ALL_ACCESS = 0xF003F
        winreg_mock.OpenKey.side_effect = PermissionError("Access denied")

        with patch.dict("sys.modules", {"winreg": winreg_mock}):
            add_to_path_windows("C:\\Some\\Folder")

    @patch.object(sys, "platform", "win32")
    @patch("definers.web.broadcast_path_change")
    def test_strips_quotes_from_folder_path(
        self, mock_broadcast: MagicMock
    ) -> None:
        mock_key = MagicMock()
        folder_with_quotes = '"C:\\New\\Folder"'
        expected_folder = os.path.normpath("C:\\New\\Folder")

        winreg_mock = MagicMock()
        winreg_mock.HKEY_CURRENT_USER = 0x80000001
        winreg_mock.KEY_ALL_ACCESS = 0xF003F
        winreg_mock.REG_EXPAND_SZ = 2
        winreg_mock.OpenKey.return_value = mock_key
        winreg_mock.QueryValueEx.return_value = ("C:\\Other", 2)

        with patch.dict("sys.modules", {"winreg": winreg_mock}):
            with patch.dict(os.environ, {"PATH": "C:\\Other"}):
                add_to_path_windows(folder_with_quotes)

                set_call_args = winreg_mock.SetValueEx.call_args
                new_path_value = set_call_args[0][4]
                self.assertTrue(new_path_value.startswith(expected_folder))


if __name__ == "__main__":
    unittest.main()
