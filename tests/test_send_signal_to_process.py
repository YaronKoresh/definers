import os
import signal
import unittest
from unittest.mock import call, patch

from definers import send_signal_to_process


class TestSendSignalToProcess(unittest.TestCase):

    @patch("os.kill")
    def test_send_signal_successfully(self, mock_os_kill):
        pid = 12345
        sig = signal.SIGTERM

        result = send_signal_to_process(pid, sig)

        self.assertTrue(result)
        mock_os_kill.assert_called_once_with(pid, sig)

    @patch("os.kill", side_effect=OSError("Test OSError"))
    def test_send_signal_failure_os_error(self, mock_os_kill):
        pid = 54321
        sig = signal.SIGKILL

        with patch("builtins.print") as mock_print:
            result = send_signal_to_process(pid, sig)

        self.assertFalse(result)
        mock_os_kill.assert_called_once_with(pid, sig)
        mock_print.assert_called_once()
        self.assertIn(
            "Error sending signal", mock_print.call_args[0][0]
        )

    @patch("os.kill")
    def test_send_signal_with_zero_pid(self, mock_os_kill):
        pid = 0
        sig = signal.SIGUSR1

        result = send_signal_to_process(pid, sig)

        self.assertTrue(result)
        mock_os_kill.assert_called_once_with(pid, sig)

    @patch("os.kill", side_effect=ProcessLookupError)
    def test_send_signal_process_not_found(self, mock_os_kill):
        pid = 99999
        sig = signal.SIGHUP

        with patch("builtins.print"):
            result = send_signal_to_process(pid, sig)

        self.assertFalse(result)
        mock_os_kill.assert_called_once_with(pid, sig)


if __name__ == "__main__":
    unittest.main()
