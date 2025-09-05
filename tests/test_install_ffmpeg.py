import unittest
from unittest.mock import patch, MagicMock
import sys

from definers import install_ffmpeg

class TestInstallFfmpeg(unittest.TestCase):

    @patch('definers.installed', return_value=True)
    def test_ffmpeg_already_installed(self, mock_installed):
        self.assertTrue(install_ffmpeg())
        mock_installed.assert_called_once_with("ffmpeg")

    @patch('definers.installed', return_value=False)
    @patch('definers.get_os_name', return_value='windows')
    @patch('definers._install_ffmpeg_windows')
    def test_install_on_windows(self, mock_install_windows, mock_get_os, mock_installed):
        self.assertTrue(install_ffmpeg())
        mock_install_windows.assert_called_once()

    @patch('definers.installed', return_value=False)
    @patch('definers.get_os_name', return_value='linux')
    @patch('definers._install_ffmpeg_linux')
    def test_install_on_linux(self, mock_install_linux, mock_get_os, mock_installed):
        self.assertTrue(install_ffmpeg())
        mock_install_linux.assert_called_once()

    @patch('definers.installed', return_value=False)
    @patch('definers.get_os_name', return_value='darwin')
    @patch('sys.exit')
    @patch('builtins.print')
    def test_unsupported_os(self, mock_print, mock_exit, mock_get_os, mock_installed):
        install_ffmpeg()
        mock_print.assert_any_call("[ERROR] Unsupported operating system: darwin.")
        mock_exit.assert_called_once_with(1)

if __name__ == '__main__':
    unittest.main()
