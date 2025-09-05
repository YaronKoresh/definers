import unittest
from unittest.mock import patch
import sys
from definers import run

class TestRun(unittest.TestCase):

    @patch('definers.run_windows')
    def test_run_on_windows(self, mock_run_windows):
        with patch.object(sys, 'platform', 'win32'):
            command = 'dir'
            silent = True
            env = {'TEST_ENV': '123'}
            run(command, silent=silent, env=env)
            mock_run_windows.assert_called_once_with(command, silent=silent, env=env)

    @patch('definers.run_linux')
    def test_run_on_linux(self, mock_run_linux):
        with patch.object(sys, 'platform', 'linux'):
            command = 'ls'
            silent = False
            env = {'TEST_ENV': 'abc'}
            run(command, silent=silent, env=env)
            mock_run_linux.assert_called_once_with(command, silent=silent, env=env)
            
    @patch('definers.run_linux')
    def test_run_on_darwin(self, mock_run_linux):
        with patch.object(sys, 'platform', 'darwin'):
            command = 'ls -l'
            run(command)
            mock_run_linux.assert_called_once_with(command, silent=False, env={})

if __name__ == '__main__':
    unittest.main()
