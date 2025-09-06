import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from definers import copy


class TestCopy(unittest.TestCase):

    @patch("os.path.isdir")
    @patch("pathlib.Path.is_symlink")
    @patch("shutil.copytree")
    @patch("shutil.copy")
    def test_copy_directory(
        self, mock_copy, mock_copytree, mock_is_symlink, mock_isdir
    ):
        mock_isdir.return_value = True
        mock_is_symlink.return_value = False

        src = "/test/src_dir"
        dst = "/test/dst_dir"

        copy(src, dst)

        mock_copytree.assert_called_once_with(
            src, dst, symlinks=False, ignore_dangling_symlinks=True
        )
        mock_copy.assert_not_called()

    @patch("os.path.isdir")
    @patch("pathlib.Path")
    @patch("shutil.copytree")
    @patch("shutil.copy")
    def test_copy_symlinked_directory(
        self, mock_copy, mock_copytree, mock_path, mock_isdir
    ):
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.is_symlink.return_value = True
        mock_path_instance.resolve.return_value = "/resolved/path"

        def isdir_side_effect(path):
            if path == "/resolved/path":
                return True
            return False

        mock_isdir.side_effect = isdir_side_effect

        src = "/test/src_symlink"
        dst = "/test/dst_dir"

        copy(src, dst)

        mock_copytree.assert_called_once_with(
            src, dst, symlinks=False, ignore_dangling_symlinks=True
        )
        mock_copy.assert_not_called()

    @patch("os.path.isdir")
    @patch("pathlib.Path")
    @patch("shutil.copytree")
    @patch("shutil.copy")
    def test_copy_file(
        self, mock_copy, mock_copytree, mock_path, mock_isdir
    ):
        mock_isdir.return_value = False
        mock_path.return_value.is_symlink.return_value = False

        src = "/test/src_file.txt"
        dst = "/test/dst_file.txt"

        copy(src, dst)

        mock_copy.assert_called_once_with(src, dst)
        mock_copytree.assert_not_called()

    @patch("os.path.isdir")
    @patch("pathlib.Path")
    @patch("shutil.copytree")
    @patch("shutil.copy")
    def test_copy_symlinked_file(
        self, mock_copy, mock_copytree, mock_path, mock_isdir
    ):
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.is_symlink.return_value = True
        mock_path_instance.resolve.return_value = "/resolved/file.txt"

        def isdir_side_effect(path):
            if path == "/test/src_symlink":
                return False
            if path == "/resolved/file.txt":
                return False
            return False

        mock_isdir.side_effect = isdir_side_effect

        src = "/test/src_symlink"
        dst = "/test/dst_dir"

        copy(src, dst)

        mock_copy.assert_called_once_with(src, dst)
        mock_copytree.assert_not_called()


if __name__ == "__main__":
    unittest.main()
