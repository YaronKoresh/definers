import unittest
from unittest.mock import ANY, patch

from definers.system import runnable


class TestRunnable(unittest.TestCase):
    @patch("shutil.which", return_value="/bin/ls")
    def test_runnable_linux_true(self, mock_which):
        self.assertTrue(runnable("ls"))
        mock_which.assert_called_once_with("ls", path=ANY)

    @patch("shutil.which", return_value=None)
    def test_runnable_linux_false(self, mock_which):
        self.assertFalse(runnable("nonexistentcommand"))
        mock_which.assert_called_once_with("nonexistentcommand", path=ANY)

    @patch("shutil.which", return_value="/bin/ls")
    def test_runnable_linux_with_args(self, mock_which):
        self.assertTrue(runnable("ls -l"))
        mock_which.assert_called_once_with("ls", path=ANY)

    @patch("shutil.which", return_value="C:\\Windows\\System32\\cmd.exe")
    def test_runnable_windows_true(self, mock_which):
        self.assertTrue(runnable("cmd"))
        mock_which.assert_called_once_with("cmd", path=ANY)

    @patch("shutil.which", return_value=None)
    def test_runnable_windows_false(self, mock_which):
        self.assertFalse(runnable("nonexistentcommand"))
        mock_which.assert_called_once_with("nonexistentcommand", path=ANY)

    @patch(
        "shutil.which",
        return_value="C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
    )
    def test_runnable_windows_with_args(self, mock_which):
        self.assertTrue(runnable("powershell.exe -Command Get-ChildItem"))
        mock_which.assert_called_once_with("powershell.exe", path=ANY)

    @patch("shutil.which")
    def test_runnable_empty_command(self, mock_which):
        self.assertFalse(runnable("   "))
        mock_which.assert_not_called()

    @patch(
        "shutil.which",
        return_value="C:\\Program Files\\Tool\\tool.exe",
    )
    def test_runnable_quoted_executable(self, mock_which):
        self.assertTrue(runnable('"C:\\Program Files\\Tool\\tool.exe" --help'))
        mock_which.assert_called_once_with(
            "C:\\Program Files\\Tool\\tool.exe",
            path=ANY,
        )


if __name__ == "__main__":
    unittest.main()
