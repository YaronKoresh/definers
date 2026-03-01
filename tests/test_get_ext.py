import unittest
from definers import get_ext


class TestGetExt(unittest.TestCase):
    def test_standard_extension(self):
        result = get_ext("/foo/bar/file.mp3")
        self.assertEqual(result, "mp3")

    def test_no_extension(self):
        result = get_ext("/foo/bar/file")
        self.assertEqual(result, "")

    def test_dot_in_path(self):
        result = get_ext("/foo.bar/file.wav")
        self.assertEqual(result, "wav")


if __name__ == "__main__":
    unittest.main()
