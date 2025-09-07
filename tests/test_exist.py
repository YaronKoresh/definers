import unittest
from unittest.mock import patch
import os
from definers import exist

class TestExist(unittest.TestCase):
    @patch("os.path.exists")
    @patch("os.path.expanduser")
    @patch("os.path.abspath")
    def test_existing_path(
        self, mock_abspath, mock_expanduser, mock_exists
    ):
        mock_exists.return_value = True
        self.assertTrue(exist("/path/to/existing_file"))

    @patch("os.path.exists")
    @patch("os.path.expanduser")
    @patch("os.path.abspath")
    def test_non_existing_path(
        self, mock_abspath, mock_expanduser, mock_exists
    ):
        mock_exists.return_value = False
        self.assertFalse(exist("/path/to/non_existing_file"))

    def test_empty_string_path(self):
        self.assertFalse(exist(""))

    def test_whitespace_path(self):
        self.assertFalse(exist("   "))

    @patch("os.path.exists")
    @patch("os.path.expanduser")
    @patch("os.path.abspath")
    def test_path_expansion(
        self, mock_abspath, mock_expanduser, mock_exists
    ):
        mock_expanduser.return_value = "/home/user/some_file"
        mock_abspath.return_value = "/home/user/some_file"
        mock_exists.return_value = True

        self.assertTrue(exist("~/some_file"))
        mock_expanduser.assert_called_once_with("~/some_file")
        mock_abspath.assert_called_once_with("/home/user/some_file")
        mock_exists.assert_called_once_with("/home/user/some_file")

if __name__ == "__main__":
    unittest.main()

