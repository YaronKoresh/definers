import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from definers import cwd


class TestCwd(unittest.TestCase):

    def setUp(self):
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)

    def test_cwd_changes_directory_and_restores(self):
        with cwd(self.temp_dir):
            self.assertEqual(os.getcwd(), self.temp_dir)
        self.assertEqual(os.getcwd(), self.original_cwd)

    def test_cwd_restores_directory_after_exception(self):
        with self.assertRaises(ValueError):
            with cwd(self.temp_dir):
                self.assertEqual(os.getcwd(), self.temp_dir)
                raise ValueError("Test exception")
        self.assertEqual(os.getcwd(), self.original_cwd)

    def test_cwd_with_non_existent_directory(self):
        non_existent_dir = os.path.join(self.temp_dir, "non_existent")
        with self.assertRaises(FileNotFoundError):
            with cwd(non_existent_dir):
                pass
        self.assertEqual(os.getcwd(), self.original_cwd)

    @patch("definers.os.path.dirname")
    def test_cwd_with_no_directory_provided(self, mock_dirname):
        mock_script_dir = "/fake/script/dir"
        mock_dirname.return_value = mock_script_dir

        with cwd():
            self.assertEqual(os.getcwd(), mock_script_dir)

        self.assertEqual(os.getcwd(), self.original_cwd)


if __name__ == "__main__":
    unittest.main()
