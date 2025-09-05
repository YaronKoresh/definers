import unittest
from unittest.mock import patch, MagicMock
import subprocess
from definers import installed

class TestInstalled(unittest.TestCase):

    @patch('definers.get_os_name', return_value='linux')
    @patch('definers.run')
    @patch('definers.shutil.which')
    def test_installed_linux_command_exists_no_version(self, mock_which, mock_run, mock_os):
        mock_which.return_value = '/usr/bin/git'
        self.assertTrue(installed('git'))
        mock_which.assert_called_once_with('git')
        mock_run.assert_not_called()

    @patch('definers.get_os_name', return_value='linux')
    @patch('definers.shutil.which')
    def test_installed_linux_command_not_exists(self, mock_which, mock_os):
        mock_which.return_value = None
        # Fallback to pip check
        with patch('definers.run') as mock_run:
            mock_run.return_value = ["packageA 1.0", "packageB 2.0"]
            self.assertFalse(installed('nonexistent'))
        mock_which.assert_called_with('nonexistent')

    @patch('definers.get_os_name', return_value='linux')
    @patch('definers.run')
    @patch('definers.shutil.which')
    def test_installed_linux_command_version_match(self, mock_which, mock_run, mock_os):
        mock_which.return_value = '/usr/bin/ffmpeg'
        mock_run.return_value = ["ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers"]
        self.assertTrue(installed('ffmpeg', '4.4.2'))
        mock_run.assert_any_call('ffmpeg --version', silent=True)
        
    @patch('definers.get_os_name', return_value='linux')
    @patch('definers.run')
    @patch('definers.shutil.which')
    def test_installed_linux_command_version_wildcard(self, mock_which, mock_run, mock_os):
        mock_which.return_value = '/usr/bin/ffmpeg'
        mock_run.return_value = ["ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers"]
        self.assertTrue(installed('ffmpeg', '4.4.*'))

    @patch('definers.get_os_name', return_value='windows')
    @patch('definers.run')
    def test_installed_windows_program_exists_no_version(self, mock_run, mock_os):
        mock_run.return_value = ["Microsoft Visual C++ 2015-2022 Redistributable (x64)  14.32.31332", "Git  2.37.1"]
        self.assertTrue(installed('git'))

    @patch('definers.get_os_name', return_value='windows')
    @patch('definers.run')
    def test_installed_windows_program_version_match(self, mock_run, mock_os):
        mock_run.return_value = ["Microsoft Visual C++ 2015-2022 Redistributable (x64)  14.32.31332", "Git  2.37.1"]
        self.assertTrue(installed('git', '2.37.1'))

    @patch('definers.get_os_name', return_value='windows')
    @patch('definers.run')
    def test_installed_windows_program_not_found(self, mock_run, mock_os):
        mock_run.side_effect = [
            ["Some Other Program 1.0"], 
            ["pip-package 1.0"] 
        ]
        self.assertFalse(installed('nonexistent'))
        self.assertEqual(mock_run.call_count, 2)

    @patch('definers.get_os_name', return_value='linux')
    @patch('definers.shutil.which', return_value=None)
    @patch('definers.run')
    def test_installed_pip_package_exists(self, mock_run, mock_which, mock_os):
        mock_run.return_value = ["requests          2.28.1", "numpy             1.21.5"]
        self.assertTrue(installed('requests'))
        self.assertTrue(installed('numpy', '1.21.5'))
        self.assertTrue(installed('numpy', '1.21.*'))
        self.assertFalse(installed('tensorflow'))

    @patch('definers.run')
    def test_installed_pip_package_version_mismatch(self, mock_run):
        mock_run.return_value = ["requests          2.28.1"]
        self.assertFalse(installed('requests', '2.29.0'))

    @patch('definers.run', side_effect=subprocess.CalledProcessError(1, 'pip'))
    def test_installed_pip_fails(self, mock_run):
        with self.assertRaises(subprocess.CalledProcessError):
             # Depending on implementation, you might want to check it returns False
             # self.assertFalse(installed('anypackage'))
             installed('anypackage')


if __name__ == '__main__':
    unittest.main()
