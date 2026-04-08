import os
import tempfile
import unittest
from importlib import import_module
from pathlib import Path


def _load_modules():
    return import_module("definers.path_utils"), import_module(
        "definers.system.paths"
    )


class TestFullPath(unittest.TestCase):
    def test_re_exports_platform_function(self):
        path_utils, platform_paths = _load_modules()
        self.assertIs(path_utils.full_path, platform_paths.full_path)

    def test_returns_resolved_absolute_path(self):
        path_utils, _ = _load_modules()
        expected = str(Path(".").expanduser().resolve())
        self.assertEqual(path_utils.full_path("."), expected)

    def test_joins_components_after_stripping_whitespace(self):
        path_utils, _ = _load_modules()
        result = path_utils.full_path("foo ", " bar", "baz.txt ")
        expected = str(
            Path(os.path.join("foo", "bar", "baz.txt")).expanduser().resolve()
        )
        self.assertEqual(result, expected)

    def test_preserves_absolute_path_resolution(self):
        path_utils, _ = _load_modules()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = path_utils.full_path(temp_dir)
            expected = str(Path(temp_dir).expanduser().resolve())
            self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
