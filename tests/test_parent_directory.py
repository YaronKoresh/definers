import os
import unittest

from definers import parent_directory


class TestParentDirectory(unittest.TestCase):
    def test_single_level(self):
        result = parent_directory("/foo/bar/baz")
        self.assertEqual(result, os.path.normpath("/foo/bar"))

    def test_multiple_levels(self):
        result = parent_directory("/foo/bar/baz", levels=2)
        self.assertEqual(result, os.path.normpath("/foo"))

    def test_zero_levels(self):
        result = parent_directory("/foo/bar", levels=0)
        self.assertEqual(result, os.path.normpath("/foo/bar"))


if __name__ == "__main__":
    unittest.main()
