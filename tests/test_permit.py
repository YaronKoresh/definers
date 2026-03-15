import os
import shutil
import stat
import tempfile
import unittest
from unittest.mock import patch

import definers.platform.filesystem as filesystem


class TestPermit(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_file = tempfile.NamedTemporaryFile(
            dir=self.test_dir, delete=False
        ).name

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch.object(filesystem, "get_os_name", return_value="linux")
    @patch.object(filesystem.subprocess, "run")
    def test_permit_calls_chmod(self, mock_subprocess_run, mock_get_os_name):
        path = self.test_file
        filesystem.permit(path, get_os_name_func=filesystem.get_os_name)
        mock_subprocess_run.assert_called_once_with(
            ["chmod", "-R", "a+xrw", path], check=True
        )

    @patch.object(filesystem.subprocess, "run", return_value=True)
    def test_permit_success(self, mock_subprocess_run):
        self.assertTrue(filesystem.permit(self.test_file))

    @patch.object(
        filesystem.subprocess,
        "run",
        side_effect=Exception("chmod failed"),
    )
    def test_permit_failure(self, mock_subprocess_run):
        self.assertFalse(filesystem.permit(self.test_file))

    def test_permit_functional_file(self):
        if os.name != "nt":
            os.chmod(self.test_file, 0)
            filesystem.permit(self.test_file)
            mode = stat.S_IMODE(os.stat(self.test_file).st_mode)
            self.assertEqual(mode, 511)

    def test_permit_functional_directory(self):
        if os.name != "nt":
            os.chmod(self.test_dir, 0)
            filesystem.permit(self.test_dir)
            mode = stat.S_IMODE(os.stat(self.test_dir).st_mode)
            self.assertEqual(mode, 511)

    def test_permit_non_existent_path(self):
        non_existent_path = os.path.join(self.test_dir, "non_existent")
        self.assertFalse(filesystem.permit(non_existent_path))


if __name__ == "__main__":
    unittest.main()
