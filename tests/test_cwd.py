import os
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from unittest.mock import patch


def _load_modules():
    return import_module("definers.path_utils"), import_module(
        "definers.platform.paths"
    )


class TestCwd(unittest.TestCase):
    def test_re_exports_platform_function(self):
        path_utils, platform_paths = _load_modules()
        self.assertIs(path_utils.cwd, platform_paths.cwd)

    @patch("os.getcwd")
    @patch("os.chdir")
    def test_cwd_with_directory_provided(self, mock_chdir, mock_getcwd):
        path_utils, _ = _load_modules()
        original_dir = str(Path(tempfile.gettempdir()).resolve())
        new_dir = os.path.join(original_dir, "new", "test", "dir")
        expected_new = str(Path(new_dir).expanduser().resolve())
        expected_owd = str(Path(original_dir).expanduser().resolve())
        mock_getcwd.return_value = original_dir
        with path_utils.cwd(new_dir):
            mock_chdir.assert_called_once_with(expected_new)
        self.assertEqual(mock_chdir.call_args_list[-1].args[0], expected_owd)
        self.assertEqual(mock_chdir.call_count, 2)

    @patch("os.getcwd")
    @patch("os.chdir")
    def test_cwd_with_no_directory_provided(self, mock_chdir, mock_getcwd):
        path_utils, platform_paths = _load_modules()
        original_dir = str(Path(tempfile.gettempdir()).resolve())
        package_root = str(
            Path(platform_paths.__file__).resolve().parent.parent
        )
        expected_new = str(
            Path(os.path.join(package_root, ".")).expanduser().resolve()
        )
        expected_owd = str(Path(original_dir).expanduser().resolve())
        mock_getcwd.return_value = original_dir
        with path_utils.cwd():
            mock_chdir.assert_called_once_with(expected_new)
        self.assertEqual(mock_chdir.call_args_list[-1].args[0], expected_owd)
        self.assertEqual(mock_chdir.call_count, 2)

    def test_cwd_actually_changes_directory(self):
        path_utils, _ = _load_modules()
        original_dir = str(Path(os.getcwd()).resolve())
        with tempfile.TemporaryDirectory() as temp_dir:
            expected_dir = str(Path(temp_dir).resolve())
            with path_utils.cwd(temp_dir):
                self.assertEqual(str(Path(os.getcwd()).resolve()), expected_dir)
            self.assertEqual(str(Path(os.getcwd()).resolve()), original_dir)


if __name__ == "__main__":
    unittest.main()
