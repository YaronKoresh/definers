import sys
import unittest
from unittest.mock import MagicMock, patch

from definers.system import is_admin_windows


class TestIsAdminWindows(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith("win"), "Windows-only test")
    @patch("ctypes.windll.shell32.IsUserAnAdmin", return_value=1, create=True)
    def test_is_admin_returns_true(self, mock_is_admin):
        self.assertTrue(is_admin_windows())

    @unittest.skipUnless(sys.platform.startswith("win"), "Windows-only test")
    @patch("ctypes.windll.shell32.IsUserAnAdmin", return_value=0, create=True)
    def test_is_not_admin_returns_false(self, mock_is_admin):
        self.assertFalse(is_admin_windows())

    def test_not_on_windows_returns_false(self):
        if sys.platform.startswith("win"):
            with patch("ctypes.windll", None, create=True):
                self.assertFalse(is_admin_windows())
        else:
            self.assertFalse(is_admin_windows())

    @unittest.skipUnless(sys.platform.startswith("win"), "Windows-only test")
    @patch(
        "ctypes.windll.shell32.IsUserAnAdmin",
        side_effect=Exception,
        create=True,
    )
    def test_ctypes_call_fails_returns_false(self, mock_is_admin):
        self.assertFalse(is_admin_windows())


if __name__ == "__main__":
    unittest.main()
