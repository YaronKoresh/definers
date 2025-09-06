import unittest
from unittest.mock import patch, MagicMock
from definers import copy

class TestCopy(unittest.TestCase):

    @patch("shutil.copy")
    @patch("shutil.copytree")
    @patch("pathlib.Path.is_symlink", return_value=False)
    @patch("os.path.isdir", return_value=False)
    def test_copy_file(
        self, mock_isdir, mock_is_symlink, mock_copytree, mock_copy
    ):
        src = "/test/src_file.txt"
        dst = "/test/dst_file.txt"
        copy(src, dst)
        mock_copy.assert_called_once_with(src, dst)
        mock_copytree.assert_not_called()

    @patch("shutil.copy")
    @patch("shutil.copytree")
    @patch("pathlib.Path.is_symlink", return_value=False)
    @patch("os.path.isdir", return_value=True)
    def test_copy_directory(
        self, mock_isdir, mock_is_symlink, mock_copytree, mock_copy
    ):
        src = "/test/src_dir"
        dst = "/test/dst_dir"
        copy(src, dst)
        mock_copytree.assert_called_once_with(
            src, dst, symlinks=False, ignore_dangling_symlinks=True
        )
        mock_copy.assert_not_called()

    @patch("shutil.copy")
    @patch("shutil.copytree")
    def test_copy_directory_with_symlink(self, mock_copytree, mock_copy):
        src = "/test/src_symlink"
        dst = "/test/dst_dir"

        with patch("pathlib.Path") as mock_path_class:
            instance = mock_path_class.return_value
            instance.is_symlink.return_value = True
            instance.resolve.return_value = "/test/resolved_dir"
            
            with patch("os.path.isdir") as mock_isdir:
                mock_isdir.side_effect = lambda path: path == "/test/resolved_dir"

                copy(src, dst)

                mock_copytree.assert_called_once_with(
                    src, dst, symlinks=False, ignore_dangling_symlinks=True
                )
                mock_copy.assert_not_called()

    @patch("shutil.copy")
    @patch("shutil.copytree")
    def test_copy_symlink_to_file(self, mock_copytree, mock_copy):
        src = "/test/src_symlink_file"
        dst = "/test/dst_file.txt"

        with patch("pathlib.Path") as mock_path_class:
            instance = mock_path_class.return_value
            instance.is_symlink.return_value = True
            instance.resolve.return_value = "/test/resolved_file.txt"
            
            with patch("os.path.isdir", return_value=False):
                copy(src, dst)
                mock_copy.assert_called_once_with(src, dst)
                mock_copytree.assert_not_called()

if __name__ == "__main__":
    unittest.main()