import os
import tempfile
import unittest
from importlib import import_module
from unittest.mock import patch


def _load_modules():
    return import_module("definers.path_utils"), import_module(
        "definers.system.paths"
    )


class TestNormalizePath(unittest.TestCase):
    def test_re_exports_platform_function(self):
        path_utils, platform_paths = _load_modules()
        self.assertIs(path_utils.normalize_path, platform_paths.normalize_path)

    def test_normalizes_relative_segments(self):
        path_utils, _ = _load_modules()
        raw_path = os.path.join("folder", "child", "..", "file.txt")
        expected = os.path.expanduser(os.path.normpath(raw_path))
        self.assertEqual(path_utils.normalize_path(raw_path), expected)

    def test_expands_user_directory_after_normalization(self):
        path_utils, _ = _load_modules()
        with tempfile.TemporaryDirectory() as home_dir:
            raw_path = os.path.join("~", "folder", "..", "file.txt")
            with patch.dict(
                os.environ,
                {"HOME": home_dir, "USERPROFILE": home_dir},
                clear=False,
            ):
                expected = os.path.expanduser(os.path.normpath(raw_path))
                self.assertEqual(path_utils.normalize_path(raw_path), expected)

    def test_normalizes_absolute_path_without_fallbacks(self):
        path_utils, _ = _load_modules()
        raw_path = os.path.join(
            tempfile.gettempdir(), "folder", ".", "child", "..", "file.txt"
        )
        expected = os.path.expanduser(os.path.normpath(raw_path))
        self.assertEqual(path_utils.normalize_path(raw_path), expected)


if __name__ == "__main__":
    unittest.main()
