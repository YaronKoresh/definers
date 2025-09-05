import unittest
from unittest.mock import patch, MagicMock
from definers import is_admin_windows

class TestIsAdminWindows(unittest.TestCase):

    @patch('ctypes.windll.shell32.IsUserAnAdmin', return_value=1)
    def test_is_admin_returns_true(self, mock_is_admin):
        self.assertTrue(is_admin_windows())

    @patch('ctypes.windll.shell32.IsUserAnAdmin', return_value=0)
    def test_is_not_admin_returns_false(self, mock_is_admin):
        self.assertFalse(is_admin_windows())

    @patch('ctypes.windll', side_effect=AttributeError)
    def test_not_on_windows_returns_false(self, mock_windll):
        self.assertFalse(is_admin_windows())

    @patch('ctypes.windll.shell32.IsUserAnAdmin', side_effect=Exception)
    def test_ctypes_call_fails_returns_false(self, mock_is_admin):
        self.assertFalse(is_admin_windows())

if __name__ == '__main__':
    unittest.main()
