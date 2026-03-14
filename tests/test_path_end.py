import unittest

from definers.path_utils import path_end


class TestPathEnd(unittest.TestCase):
    def test_simple_path(self):
        result = path_end("/foo/bar/baz.txt")
        self.assertEqual(result, "baz.txt")

    def test_directory_path(self):
        result = path_end("/foo/bar/")
        self.assertIn(result, ["bar", ""])

    def test_single_component(self):
        result = path_end("file.py")
        self.assertEqual(result, "file.py")


if __name__ == "__main__":
    unittest.main()
