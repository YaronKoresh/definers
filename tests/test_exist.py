import unittest
from unittest.mock import patch

from definers import exist


class TestExist(unittest.TestCase):

    @patch("definers.run")
    def test_existing_file(self, mock_run):
        mock_run.return_value = ["/path/to/existing_file.txt"]
        self.assertTrue(exist("/path/to/existing_file.txt"))
        mock_run.assert_called_with(
            'ls -1 "/path/to/existing_file.txt"', silent=True
        )

    @patch("definers.run")
    def test_existing_directory(self, mock_run):
        mock_run.return_value = ["/path/to/existing_dir/"]
        self.assertTrue(exist("/path/to/existing_dir/"))
        mock_run.assert_called_with(
            'ls -1 "/path/to/existing_dir"', silent=True
        )

    @patch("definers.run")
    def test_non_existing_path(self, mock_run):
        mock_run.return_value = False
        self.assertFalse(exist("/path/to/non_existing_file.txt"))
        mock_run.assert_called_with(
            'ls -1 "/path/to/non_existing_file.txt"', silent=True
        )

    @patch("definers.run")
    def test_path_with_spaces(self, mock_run):
        mock_run.return_value = ["/path/to/a file with spaces.txt"]
        self.assertTrue(exist("/path/to/a file with spaces.txt"))
        mock_run.assert_called_with(
            'ls -1 "/path/to/a file with spaces.txt"', silent=True
        )

    @patch("definers.run")
    def test_empty_string_path(self, mock_run):
        mock_run.return_value = False
        self.assertFalse(exist(""))

    @patch("definers.run")
    def test_whitespace_path(self, mock_run):
        mock_run.return_value = False
        self.assertFalse(exist("   "))

    @patch("os.path.abspath", return_value="/abs/path")
    @patch("os.path.expanduser", return_value="/home/user")
    @patch("definers.run")
    def test_path_expansion(
        self, mock_run, mock_expanduser, mock_abspath
    ):
        mock_run.return_value = ["/home/user/some_file"]
        exist("~/some_file")
        mock_expanduser.assert_called_with("~/some_file")
        mock_abspath.assert_called_with("/home/user/some_file")


if __name__ == "__main__":
    unittest.main()
