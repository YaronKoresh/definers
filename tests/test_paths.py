import unittest
from unittest.mock import call, patch

from definers import paths


class TestPaths(unittest.TestCase):
    def test_no_patterns_provided(self):
        result = paths()
        self.assertEqual(result, [])

    @patch("definers.glob")
    @patch("os.path.abspath")
    @patch("os.path.expanduser")
    def test_duplicate_paths_are_removed(
        self, mock_expanduser, mock_abspath, mock_glob
    ):
        mock_abspath.side_effect = lambda p: p
        mock_expanduser.side_effect = lambda p: p
        mock_glob.side_effect = [
            ["/data/file.csv"],
            ["/data/file.csv", "/data/another.csv"],
        ]
        result = paths("/data/file.csv", "/data/*.csv")
        self.assertCountEqual(result, ["/data/file.csv", "/data/another.csv"])

    @patch("definers.glob")
    @patch("os.path.abspath")
    @patch("os.path.expanduser")
    def test_glob_exception(self, mock_expanduser, mock_abspath, mock_glob):
        mock_abspath.side_effect = lambda p: p
        mock_expanduser.side_effect = lambda p: p
        mock_glob.side_effect = Exception("Test exception")
        result = paths("/some/pattern/*")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
