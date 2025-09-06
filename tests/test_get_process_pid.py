import subprocess
import unittest
from unittest.mock import patch

from definers import get_process_pid


class TestGetProcessPid(unittest.TestCase):

    @patch("subprocess.check_output")
    def test_get_process_pid_success(self, mock_check_output):
        mock_check_output.return_value = b"12345\n"
        pid = get_process_pid("some_process")
        self.assertEqual(pid, 12345)
        mock_check_output.assert_called_once_with(
            ["pidof", "some_process"]
        )

    @patch("subprocess.check_output")
    def test_get_process_pid_not_found(self, mock_check_output):
        mock_check_output.side_effect = subprocess.CalledProcessError(
            1, "pidof"
        )
        pid = get_process_pid("nonexistent_process")
        self.assertIsNone(pid)

    @patch("subprocess.check_output")
    def test_get_process_pid_invalid_output(self, mock_check_output):
        mock_check_output.return_value = b"not a pid\n"
        pid = get_process_pid("some_process")
        self.assertIsNone(pid)

    @patch("subprocess.check_output")
    def test_get_process_pid_multiple_pids(self, mock_check_output):
        mock_check_output.return_value = b"12345 67890\n"
        pid = get_process_pid("some_process")
        self.assertIsNone(pid)

    @patch("subprocess.check_output")
    def test_get_process_pid_empty_output(self, mock_check_output):
        mock_check_output.return_value = b""
        pid = get_process_pid("some_process")
        self.assertIsNone(pid)


if __name__ == "__main__":
    unittest.main()
