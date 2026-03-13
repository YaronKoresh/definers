import sys
import unittest
from unittest.mock import MagicMock, patch

from definers import run_linux


class TestRunLinux(unittest.TestCase):
    @unittest.skipIf(sys.platform.startswith("win"), "Linux-specific test")
    @patch("subprocess.Popen")
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

    @patch("definers.log")
    def test_run_linux_rejects_unsafe_string(self, mock_log):

        result = run_linux("echo hi; rm -rf /")
        self.assertFalse(result)

    @unittest.skipIf(sys.platform.startswith("win"), "Linux-specific test")
    @patch("subprocess.Popen")
    def test_run_linux_failure(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("", "error")
        mock_proc.returncode = 1
        mock_popen.return_value = mock_proc
        result = run_linux("exit 1", silent=True)
        self.assertFalse(result)

    @patch("definers.run_linux", return_value=["/usr/bin"])
    def test_run_linux_with_env(self, mock_run_linux):
        if sys.platform.startswith("win"):
            self.skipTest("Linux-specific test")
        run_linux("echo $MY_VAR", env={"MY_VAR": "test_value"})
        pass

    def test_run_on_windows(self):
        if not sys.platform.startswith("win"):
            self.skipTest("Windows-specific check")
        pass

    @unittest.skipIf(sys.platform.startswith("win"), "Linux-specific test")
    def test_run_linux_empty_command_returns_false(self):
        self.assertFalse(run_linux("   ", silent=True))


if __name__ == "__main__":
    unittest.main()
