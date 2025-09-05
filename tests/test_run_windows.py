import unittest
from unittest.mock import patch, MagicMock
import os
import sys
from definers import run_windows

class TestRunWindows(unittest.TestCase):

    def setUp(self):
        if not sys.platform.startswith('win'):
            self.skipTest("Windows-specific tests")

    @patch('subprocess.Popen')
    def test_run_windows_success_string_command(self, mock_popen):
        mock_process = MagicMock()
        mock_process.communicate.return_value = ('success output\r\n', '')
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        result = run_windows('echo "success output"', silent=True)
        self.assertEqual(result, ['success output'])
        mock_popen.assert_called_once_with(
            'echo "success output"',
            shell=True,
            stdin=unittest.mock.ANY,
            stdout=unittest.mock.ANY,
            stderr=unittest.mock.ANY,
            env=unittest.mock.ANY,
            universal_newlines=True
        )

    @patch('subprocess.Popen')
    def test_run_windows_success_list_command(self, mock_popen):
        mock_process = MagicMock()
        mock_process.communicate.return_value = ('line1\r\nline2\r\n', '')
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        result = run_windows(['echo line1', 'echo line2'], silent=True)
        self.assertEqual(result, ['line1', 'line2'])
        mock_popen.assert_called_once_with(
            'echo line1 && echo line2',
            shell=True,
            stdin=unittest.mock.ANY,
            stdout=unittest.mock.ANY,
            stderr=unittest.mock.ANY,
            env=unittest.mock.ANY,
            universal_newlines=True
        )

    @patch('subprocess.Popen')
    def test_run_windows_failure(self, mock_popen):
        mock_process = MagicMock()
        mock_process.communicate.return_value = ('', 'error message')
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        result = run_windows('exit 1', silent=True)
        self.assertFalse(result)

    @patch('subprocess.Popen')
    def test_run_windows_with_env(self, mock_popen):
        mock_process = MagicMock()
        mock_process.communicate.return_value = ('test_value\r\n', '')
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        custom_env = {'MY_VAR': 'test_value'}
        run_windows('echo %MY_VAR%', silent=True, env=custom_env)

        expected_env = {**os.environ, **custom_env}
        
        mock_popen.assert_called_once()
        called_args, called_kwargs = mock_popen.call_args
        self.assertIn('env', called_kwargs)
        self.assertEqual(called_kwargs['env'], expected_env)

    @patch('builtins.print')
    @patch('subprocess.Popen')
    def test_run_windows_silent_mode(self, mock_popen, mock_print):
        mock_process = MagicMock()
        mock_process.communicate.return_value = ('output', 'error')
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        run_windows('some command', silent=True)
        mock_print.assert_not_called()

    @patch('builtins.print')
    @patch('subprocess.Popen')
    def test_run_windows_verbose_mode(self, mock_popen, mock_print):
        mock_process = MagicMock()
        mock_process.communicate.return_value = ('output\n', 'error\n')
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        run_windows('some command', silent=False)
        
        self.assertIn(unittest.mock.call('output\n', end='', flush=True), mock_print.call_args_list)
        self.assertIn(unittest.mock.call('error\n', end='', flush=True), mock_print.call_args_list)


if __name__ == '__main__':
    unittest.main()
