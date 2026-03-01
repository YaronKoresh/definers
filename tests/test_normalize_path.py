import os
import unittest
from definers import normalize_path


class TestNormalizePath(unittest.TestCase):
    def test_normalizes_slashes(self):
        result = normalize_path("/foo//bar/../baz")
        self.assertEqual(result, os.path.normpath("/foo//bar/../baz"))

    def test_simple_path(self):
        result = normalize_path("/home/user/file.txt")
        self.assertEqual(result, os.path.normpath("/home/user/file.txt"))

    def test_relative_path(self):
        result = normalize_path("foo/bar")
        self.assertEqual(result, os.path.normpath("foo/bar"))


if __name__ == "__main__":
    unittest.main()
