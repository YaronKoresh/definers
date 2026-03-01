import os
import tempfile
import unittest
from definers import is_symlink


class TestIsSymlink(unittest.TestCase):
    def test_not_symlink(self):
        with tempfile.NamedTemporaryFile() as f:
            self.assertFalse(is_symlink(f.name))

    def test_directory_not_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(is_symlink(d))

    def test_nonexistent_not_symlink(self):
        self.assertFalse(is_symlink("/nonexistent/path/xyz"))


if __name__ == "__main__":
    unittest.main()
