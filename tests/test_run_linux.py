import unittest
from unittest.mock import patch, MagicMock
import os
import sys
from definers import run_linux

class TestRunLinux(unittest.TestCase):

    @patch('definers.write')
    @patch('definers.permit')
    @patch('definers.delete')
    @patch('pty.openpty')
    @patch('os.fork')
    @patch('os.waitpid')
    @patch('select.select')
    @patch('os.read')
    def test_run_linux_success(self, mock_read, mock_select, mock_waitpid, mock_fork, mock_openpty, mock_delete, mock_permit, mock_write):
        if sys.platform.startswith('win'):
            self.skipTest("Linux-specific test")

        mock_openpty.return_value = (10, 20)
        mock_fork.return_value = 1234
        mock_waitpid.return_value = (1234, 0)
        
        output_stream = [b'line 1\n', b'line 2\n', b'']
        mock_read.side_effect = output_stream
        
        mock_select.return_value = ([10], [], [])

        result = run_linux("echo 'line 1' && echo 'line 2'", silent=True)
        
        self.assertEqual(result, ['line 1', 'line 2'])
        mock_write.assert_called_once()
        mock_permit.assert_called_once()
        mock_delete.assert_called_once()

    @patch('definers.write')
    @patch('definers.permit')
    @patch('definers.delete')
    @patch('pty.openpty')
    @patch('os.fork')
    @patch('os.waitpid')
    def test_run_linux_failure(self, mock_waitpid, mock_fork, mock_openpty, mock_delete, mock_permit, mock_write):
        if sys.platform.startswith('win'):
            self.skipTest("Linux-specific test")

        mock_openpty.return_value = (10, 20)
        mock_fork.return_value = 1234
        mock_waitpid.return_value = (1234, 256) 

        result = run_linux("exit 1", silent=True)
        self.assertFalse(result)

    @patch('definers.run_linux', return_value=['/usr/bin'])
    def test_run_linux_with_env(self, mock_run_linux):
        if sys.platform.startswith('win'):
            self.skipTest("Linux-specific test")
        
        run_linux('echo $MY_VAR', env={'MY_VAR': 'test_value'})
        
        pass

    def test_run_on_windows(self):
        if not sys.platform.startswith('win'):
            self.skipTest("Windows-specific check")
        
        pass

if __name__ == '__main__':
    unittest.main()
