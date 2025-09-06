import unittest
from unittest.mock import patch

from definers import tmp


class TestTmp(unittest.TestCase):

    @patch("tempfile.TemporaryDirectory")
    @patch("definers.delete")
    def test_tmp_directory_keep(self, mock_delete, mock_tempdir):
        mock_tempdir.return_value.__enter__.return_value = (
            "/tmp/test_dir"
        )
        result = tmp(dir=True, keep=True)
        self.assertEqual(result, "/tmp/test_dir")
        mock_delete.assert_not_called()

    @patch("tempfile.TemporaryDirectory")
    @patch("definers.delete")
    def test_tmp_directory_no_keep(self, mock_delete, mock_tempdir):
        mock_tempdir.return_value.__enter__.return_value = (
            "/tmp/test_dir_to_delete"
        )
        result = tmp(dir=True, keep=False)
        self.assertEqual(result, "/tmp/test_dir_to_delete")
        mock_delete.assert_called_once_with("/tmp/test_dir_to_delete")

    @patch("tempfile.NamedTemporaryFile")
    @patch("definers.delete")
    def test_tmp_file_default_keep(self, mock_delete, mock_tempfile):
        mock_tempfile.return_value.__enter__.return_value.name = (
            "/tmp/test_file.data"
        )
        result = tmp()
        self.assertEqual(result, "/tmp/test_file.data")
        mock_tempfile.assert_called_with(suffix=".data", delete=False)
        mock_delete.assert_not_called()

    @patch("tempfile.NamedTemporaryFile")
    @patch("definers.delete")
    def test_tmp_file_no_keep(self, mock_delete, mock_tempfile):
        mock_tempfile.return_value.__enter__.return_value.name = (
            "/tmp/test_file_to_delete.data"
        )
        result = tmp(keep=False)
        self.assertEqual(result, "/tmp/test_file_to_delete.data")
        mock_tempfile.assert_called_with(suffix=".data", delete=False)
        mock_delete.assert_called_once_with(
            "/tmp/test_file_to_delete.data"
        )

    @patch("tempfile.NamedTemporaryFile")
    def test_tmp_file_custom_suffix_with_dot(self, mock_tempfile):
        mock_tempfile.return_value.__enter__.return_value.name = (
            "/tmp/test_file.txt"
        )
        result = tmp(suffix=".txt")
        self.assertEqual(result, "/tmp/test_file.txt")
        mock_tempfile.assert_called_with(suffix=".txt", delete=False)

    @patch("tempfile.NamedTemporaryFile")
    def test_tmp_file_custom_suffix_without_dot(self, mock_tempfile):
        mock_tempfile.return_value.__enter__.return_value.name = (
            "/tmp/test_file.log"
        )
        result = tmp(suffix="log")
        self.assertEqual(result, "/tmp/test_file.log")
        mock_tempfile.assert_called_with(suffix=".log", delete=False)

    @patch("tempfile.NamedTemporaryFile")
    def test_tmp_file_suffix_with_multiple_dots(self, mock_tempfile):
        mock_tempfile.return_value.__enter__.return_value.name = (
            "/tmp/test_file.zip"
        )
        result = tmp(suffix="archive.zip")
        self.assertEqual(result, "/tmp/test_file.zip")
        mock_tempfile.assert_called_with(suffix=".zip", delete=False)

    @patch("tempfile.NamedTemporaryFile")
    def test_tmp_file_suffix_ending_with_dot(self, mock_tempfile):
        mock_tempfile.return_value.__enter__.return_value.name = (
            "/tmp/test_file.tmp"
        )
        result = tmp(suffix="filename.")
        self.assertEqual(result, "/tmp/test_file.tmp")
        mock_tempfile.assert_called_with(suffix=".tmp", delete=False)


if __name__ == "__main__":
    unittest.main()
