import os
import unittest
from importlib import import_module


def _load_modules():
    return import_module("definers.path_utils"), import_module(
        "definers.system.paths"
    )


class TestParentDirectory(unittest.TestCase):
    def test_re_exports_platform_function(self):
        path_utils, platform_paths = _load_modules()
        self.assertIs(
            path_utils.parent_directory, platform_paths.parent_directory
        )

    def test_single_level(self):
        path_utils, _ = _load_modules()
        target = os.path.join("alpha", "beta", "gamma")
        result = path_utils.parent_directory(target)
        self.assertEqual(
            result, os.path.normpath(os.path.join("alpha", "beta"))
        )

    def test_multiple_levels(self):
        path_utils, _ = _load_modules()
        target = os.path.join("alpha", "beta", "gamma")
        result = path_utils.parent_directory(target, levels=2)
        self.assertEqual(result, os.path.normpath("alpha"))

    def test_zero_levels(self):
        path_utils, _ = _load_modules()
        target = os.path.join("alpha", "beta")
        result = path_utils.parent_directory(target, levels=0)
        self.assertEqual(result, os.path.normpath(target))


if __name__ == "__main__":
    unittest.main()
