import os
import tempfile
import unittest
from importlib import import_module
from unittest.mock import patch


def _load_modules():
    return import_module("definers.path_utils"), import_module(
        "definers.platform.paths"
    )


class TestTmp(unittest.TestCase):
    def test_re_exports_platform_function(self):
        path_utils, platform_paths = _load_modules()
        self.assertIs(path_utils.tmp, platform_paths.tmp)

    @patch(
        "definers.platform.paths.tempfile.mkdtemp",
        return_value=os.path.join(tempfile.gettempdir(), "test_dir"),
    )
    def test_tmp_directory_keep(self, mock_mkdtemp):
        path_utils, _ = _load_modules()
        expected = os.path.join(tempfile.gettempdir(), "test_dir")
        result = path_utils.tmp(dir=True, keep=True)
        self.assertEqual(result, expected)
        mock_mkdtemp.assert_called_once()

    @patch(
        "definers.platform.paths.tempfile.mkdtemp",
        return_value=os.path.join(tempfile.gettempdir(), "test_dir_to_delete"),
    )
    @patch("definers.platform.paths.os.path.isdir", return_value=True)
    @patch("definers.platform.paths.shutil.rmtree")
    def test_tmp_directory_no_keep(self, mock_rmtree, mock_isdir, mock_mkdtemp):
        path_utils, _ = _load_modules()
        expected = os.path.join(tempfile.gettempdir(), "test_dir_to_delete")
        result = path_utils.tmp(dir=True, keep=False)
        self.assertEqual(result, expected)
        mock_mkdtemp.assert_called_once()
        mock_isdir.assert_called_once_with(expected)
        mock_rmtree.assert_called_once_with(expected, ignore_errors=True)

    @patch("definers.platform.paths.tempfile.NamedTemporaryFile")
    def test_tmp_file_default_keep(self, mock_tempfile):
        path_utils, _ = _load_modules()
        expected = os.path.join(tempfile.gettempdir(), "test_file.data")
        mock_tempfile.return_value.__enter__.return_value.name = expected
        result = path_utils.tmp()
        self.assertEqual(result, expected)
        mock_tempfile.assert_called_with(suffix=".data", delete=False)

    @patch("definers.platform.paths.tempfile.NamedTemporaryFile")
    @patch("definers.platform.paths.os.remove")
    def test_tmp_file_no_keep(self, mock_remove, mock_tempfile):
        path_utils, _ = _load_modules()
        expected = os.path.join(
            tempfile.gettempdir(), "test_file_to_delete.data"
        )
        mock_tempfile.return_value.__enter__.return_value.name = expected
        result = path_utils.tmp(keep=False)
        self.assertEqual(result, expected)
        mock_tempfile.assert_called_with(suffix=".data", delete=False)
        mock_remove.assert_called_once_with(expected)

    @patch("definers.platform.paths.tempfile.NamedTemporaryFile")
    def test_tmp_file_custom_suffix_with_dot(self, mock_tempfile):
        path_utils, _ = _load_modules()
        expected = os.path.join(tempfile.gettempdir(), "test_file.txt")
        mock_tempfile.return_value.__enter__.return_value.name = expected
        result = path_utils.tmp(suffix=".txt")
        self.assertEqual(result, expected)
        mock_tempfile.assert_called_with(suffix=".txt", delete=False)

    @patch("definers.platform.paths.tempfile.NamedTemporaryFile")
    def test_tmp_file_custom_suffix_without_dot(self, mock_tempfile):
        path_utils, _ = _load_modules()
        expected = os.path.join(tempfile.gettempdir(), "test_file.tmp")
        mock_tempfile.return_value.__enter__.return_value.name = expected
        result = path_utils.tmp(suffix="tmp")
        self.assertEqual(result, expected)
        mock_tempfile.assert_called_with(suffix=".tmp", delete=False)

    def test_tmp_file_rejects_invalid_suffix(self):
        path_utils, _ = _load_modules()
        with self.assertRaises(ValueError):
            path_utils.tmp(suffix="exe")


if __name__ == "__main__":
    unittest.main()
