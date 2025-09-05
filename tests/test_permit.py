import unittest
from unittest.mock import patch
import os
import stat
import tempfile
import shutil
from definers import permit

class TestPermit(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_file = tempfile.NamedTemporaryFile(dir=self.test_dir, delete=False).name

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('definers.subprocess.run')
    def test_permit_calls_chmod(self, mock_subprocess_run):
        path = self.test_file
        permit(path)
        mock_subprocess_run.assert_called_once_with(["chmod", "-R", "a+xrw", path], check=True)

    @patch('definers.subprocess.run', return_value=True)
    def test_permit_success(self, mock_subprocess_run):
        self.assertTrue(permit(self.test_file))

    @patch('definers.subprocess.run', side_effect=Exception("chmod failed"))
    def test_permit_failure(self, mock_subprocess_run):
        self.assertFalse(permit(self.test_file))

    def test_permit_functional_file(self):
        if os.name != 'nt':
            os.chmod(self.test_file, 0)
            permit(self.test_file)
            mode = stat.S_IMODE(os.stat(self.test_file).st_mode)
            self.assertEqual(mode, 0o777)

    def test_permit_functional_directory(self):
        if os.name != 'nt':
            os.chmod(self.test_dir, 0)
            permit(self.test_dir)
            mode = stat.S_IMODE(os.stat(self.test_dir).st_mode)
            self.assertEqual(mode, 0o777)
            
    def test_permit_non_existent_path(self):
        non_existent_path = os.path.join(self.test_dir, "non_existent")
        self.assertFalse(permit(non_existent_path))


if __name__ == '__main__':
    unittest.main()
