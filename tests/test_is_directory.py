import os
import tempfile
import unittest
from definers import is_directory


class TestIsDirectory(unittest.TestCase):
    def test_existing_directory(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(is_directory(d))

    def test_non_existing_path(self):
        self.assertFalse(is_directory("/non/existent/path/xyz123"))

    def test_file_is_not_directory(self):
        with tempfile.NamedTemporaryFile() as f:
            self.assertFalse(is_directory(f.name))


if __name__ == "__main__":
    unittest.main()
