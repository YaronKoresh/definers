import sys
import unittest
from unittest.mock import MagicMock, patch

from definers.web import broadcast_path_change


class TestBroadcastPathChange(unittest.TestCase):
    @patch.object(sys, "platform", "linux")
    def test_noop_on_linux(self) -> None:
        broadcast_path_change()

    @patch.object(sys, "platform", "darwin")
    def test_noop_on_macos(self) -> None:
        broadcast_path_change()

    @patch.object(sys, "platform", "win32")
    def test_calls_send_message_on_windows(self) -> None:
        mock_dword_instance = MagicMock()
        mock_wintypes = MagicMock()
        mock_wintypes.DWORD.return_value = mock_dword_instance

        mock_ctypes = MagicMock()
        mock_ctypes.byref.return_value = "byref_result"

        mock_send_message = MagicMock()
        mock_ctypes.windll.user32.SendMessageTimeoutW = mock_send_message

        with patch.dict(
            "sys.modules",
            {"ctypes": mock_ctypes, "ctypes.wintypes": mock_wintypes},
        ):
            broadcast_path_change()

            mock_send_message.assert_called_once_with(
                65535, 26, 0, "Environment", 2, 5000, "byref_result"
            )


if __name__ == "__main__":
    unittest.main()
