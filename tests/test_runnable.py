import unittest
from unittest.mock import patch
from definers import runnable

class TestRunnable(unittest.TestCase):

    @patch('definers.get_os_name', return_value='linux')
    @patch('definers.run')
    def test_runnable_linux_true(self, mock_run, mock_os):
        mock_run.return_value = ['/bin/ls']
        self.assertTrue(runnable('ls'))
        mock_run.assert_called_once_with("which 'ls'", silent=True)

    @patch('definers.get_os_name', return_value='linux')
    @patch('definers.run')
    def test_runnable_linux_false(self, mock_run, mock_os):
        mock_run.return_value = False
        self.assertFalse(runnable('nonexistentcommand'))
        mock_run.assert_called_once_with("which 'nonexistentcommand'", silent=True)

    @patch('definers.get_os_name', return_value='linux')
    @patch('definers.run')
    def test_runnable_linux_with_args(self, mock_run, mock_os):
        mock_run.return_value = ['/bin/ls']
        self.assertTrue(runnable('ls -l'))
        mock_run.assert_called_once_with("which 'ls'", silent=True)

    @patch('definers.get_os_name', return_value='windows')
    @patch('definers.run')
    def test_runnable_windows_true(self, mock_run, mock_os):
        mock_run.return_value = True
        self.assertTrue(runnable('cmd'))
        mock_run.assert_called_once_with("powershell.exe -Command 'cmd' -WhatIf", silent=True)

    @patch('definers.get_os_name', return_value='windows')
    @patch('definers.run')
    def test_runnable_windows_false(self, mock_run, mock_os):
        mock_run.return_value = False
        self.assertFalse(runnable('nonexistentcommand'))
        mock_run.assert_called_once_with("powershell.exe -Command 'nonexistentcommand' -WhatIf", silent=True)

    @patch('definers.get_os_name', return_value='windows')
    @patch('definers.run')
    def test_runnable_windows_with_args(self, mock_run, mock_os):
        mock_run.return_value = True
        self.assertTrue(runnable('powershell.exe -Command Get-ChildItem'))
        mock_run.assert_called_once_with("powershell.exe -Command 'powershell.exe' -WhatIf", silent=True)

if __name__ == '__main__':
    unittest.main()
