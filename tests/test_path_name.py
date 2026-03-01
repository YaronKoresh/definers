import unittest

from definers import path_name


class TestPathName(unittest.TestCase):
    def test_with_extension(self):
        result = path_name("/foo/bar/file.txt")
        self.assertEqual(result, "file")

    def test_no_extension(self):
        result = path_name("/foo/bar/file")
        self.assertEqual(result, "file")

    def test_multiple_dots(self):
        result = path_name("/foo/bar/archive.tar.gz")
        self.assertEqual(result, "archive.tar")


if __name__ == "__main__":
    unittest.main()
