import unittest
from unittest.mock import patch, MagicMock
import os
from definers import cwd

class TestCwd(unittest.TestCase):

    @patch("definers.os.getcwd", return_value="/original/path")
    @patch("definers.os.chdir")
    def test_cwd_with_directory_provided(self, mock_chdir, mock_getcwd):
        new_dir = "/new/test/dir"
        
        with cwd(new_dir):
            mock_chdir.assert_called_once_with(new_dir)
            
        mock_chdir.assert_called_with("/original/path")
        self.assertEqual(mock_chdir.call_count, 2)

    @patch("definers.os.getcwd", return_value="/original/path")
    @patch("definers.os.chdir")
    @patch("definers.os.path.dirname")
    def test_cwd_with_no_directory_provided(self, mock_dirname, mock_chdir, mock_getcwd):
        mock_script_dir = "/fake/script/dir"
        mock_dirname.return_value = mock_script_dir

        with cwd():
            mock_chdir.assert_called_once_with(mock_script_dir)
            
        mock_chdir.assert_called_with("/original/path")
        self.assertEqual(mock_chdir.call_count, 2)
        
    def test_cwd_actually_changes_directory(self):
        original_dir = os.getcwd()
        temp_dir = os.path.realpath(os.path.join(original_dir, "temp_test_dir_for_cwd"))
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            with cwd(temp_dir):
                self.assertEqual(os.getcwd(), temp_dir)
            self.assertEqual(os.getcwd(), original_dir)
        finally:
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

if __name__ == "__main__":
    unittest.main()