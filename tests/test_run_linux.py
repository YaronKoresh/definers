import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import definers.platform.processes as processes
from definers.system import run_linux


class TestRunLinux(unittest.TestCase):
    @unittest.skipIf(sys.platform.startswith("win"), "Linux-specific test")
    @patch.object(processes.subprocess, "Popen")
    def test_run_linux_success(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("line 1\nline 2\n", "")
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc
        result = run_linux("echo 'line 1'", silent=True)
        self.assertEqual(result, ["line 1", "line 2"])

    @unittest.skipIf(sys.platform.startswith("win"), "Linux-specific test")
    def test_run_linux_list_invocation(self):
        result = run_linux(["echo", "hello"], silent=True)
        self.assertEqual(result, ["hello"])

    def test_run_linux_rejects_unsafe_string(self):
        result = run_linux("echo hi; rm -rf /")
        self.assertFalse(result)

    @unittest.skipIf(sys.platform.startswith("win"), "Linux-specific test")
    @patch.object(processes.subprocess, "Popen")
    def test_run_linux_failure(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("", "error")
        mock_proc.returncode = 1
        mock_popen.return_value = mock_proc
        result = run_linux("exit 1", silent=True)
        self.assertFalse(result)

    @patch.object(processes.subprocess, "Popen")
    def test_run_linux_with_env(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("test_value\n", "")
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc
        if sys.platform.startswith("win"):
            self.skipTest("Linux-specific test")
        custom_env = {"MY_VAR": "test_value"}
        run_linux(
            ["/bin/sh", "-lc", 'printf %s "$MY_VAR"'],
            silent=True,
            env=custom_env,
        )
        (_, called_kwargs) = mock_popen.call_args
        self.assertEqual(called_kwargs["env"], {**os.environ, **custom_env})

    def test_run_on_windows(self):
        if not sys.platform.startswith("win"):
            self.skipTest("Windows-specific check")
        self.assertFalse(run_linux("echo hello", silent=True))

    @unittest.skipIf(sys.platform.startswith("win"), "Linux-specific test")
    def test_run_linux_empty_command_returns_false(self):
        self.assertFalse(run_linux("   ", silent=True))


if __name__ == "__main__":
    unittest.main()
