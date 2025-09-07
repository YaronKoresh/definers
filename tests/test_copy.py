import os
import shutil
import unittest
from unittest.mock import patch

from definers import copy


class TestCopy(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_copy_temp"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch("shutil.copy")
    @patch("shutil.copytree")
    @patch("os.path.isdir", return_value=True)
    @patch("pathlib.Path.is_symlink", return_value=False)
    def test_copy_directory(
        self, mock_is_symlink, mock_isdir, mock_copytree, mock_copy
    ):
        src = os.path.join(self.test_dir, "src_dir")
        dst = os.path.join(self.test_dir, "dst_dir")
        copy(src, dst)
        mock_copytree.assert_called_once_with(
            src, dst, symlinks=False, ignore_dangling_symlinks=True
        )
        mock_copy.assert_not_called()

    @patch("shutil.copy")
    @patch("shutil.copytree")
    @patch("os.path.isdir")
    @patch("pathlib.Path.resolve")
    @patch("pathlib.Path.is_symlink", return_value=True)
    def test_copy_directory_with_symlink(
        self,
        mock_is_symlink,
        mock_resolve,
        mock_isdir,
        mock_copytree,
        mock_copy,
    ):
        src = "/test/src_symlink"
        dst = "/test/dst_dir"
        resolved_path = "/test/resolved_dir"
        mock_resolve.return_value = resolved_path
        # When checking the resolved path of the symlink, it is a directory
        mock_isdir.side_effect = lambda path: path in [resolved_path]

        copy(src, dst)

        mock_copytree.assert_called_once_with(
            src, dst, symlinks=False, ignore_dangling_symlinks=True
        )
        mock_copy.assert_not_called()
        mock_isdir.assert_any_call(str(resolved_path))

    @patch("shutil.copy")
    @patch("shutil.copytree")
    @patch("os.path.isdir", return_value=False)
    @patch("pathlib.Path.is_symlink", return_value=False)
    def test_copy_file(
        self, mock_is_symlink, mock_isdir, mock_copytree, mock_copy
    ):
        src = os.path.join(self.test_dir, "src_file.txt")
        dst = os.path.join(self.test_dir, "dst_file.txt")
        copy(src, dst)
        mock_copy.assert_called_once_with(src, dst)
        mock_copytree.assert_not_called()

    @patch("shutil.copy")
    @patch("shutil.copytree")
    @patch("os.path.isdir", return_value=False)
    @patch("pathlib.Path.resolve")
    @patch("pathlib.Path.is_symlink", return_value=True)
    def test_copy_symlink_to_file(
        self,
        mock_is_symlink,
        mock_resolve,
        mock_isdir,
        mock_copytree,
        mock_copy,
    ):
        src = "/test/src_symlink_file"
        dst = "/test/dst_file.txt"
        resolved_path = "/test/resolved_file.txt"
        mock_resolve.return_value = resolved_path

        copy(src, dst)

        mock_copy.assert_called_once_with(src, dst)
        mock_copytree.assert_not_called()
        mock_isdir.assert_any_call(str(resolved_path))


if __name__ == "__main__":
    unittest.main()
