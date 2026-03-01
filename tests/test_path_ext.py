import unittest

from definers import path_ext


class TestPathExt(unittest.TestCase):
    def test_standard_extension(self):
        result = path_ext("/foo/bar/file.txt")
        self.assertIn("txt", result)

    def test_no_extension(self):
        result = path_ext("/foo/bar/file")
        self.assertEqual(result, "")

    def test_multiple_dots(self):
        result = path_ext("/foo/bar/archive.tar.gz")
        self.assertIn("gz", result)


if __name__ == "__main__":
    unittest.main()
