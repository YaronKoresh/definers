import subprocess
import unittest
from unittest.mock import call, patch

from definers import remove_silence


class TestRemoveSilence(unittest.TestCase):

    @patch("subprocess.run")
    def test_remove_silence_success(self, mock_run):
        input_file = "input.wav"
        output_file = "output.wav"

        result = remove_silence(input_file, output_file)

        self.assertEqual(result, output_file)

        expected_command = [
            "ffmpeg",
            "-y",
            "-i",
            input_file,
            "-ac",
            "2",
            "-af",
            "silenceremove=stop_duration=0.1:stop_threshold=-32dB",
            output_file,
        ]

        mock_run.assert_called_once_with(expected_command, check=True)

    @patch("subprocess.run")
    @patch("definers.catch")
    def test_remove_silence_failure(self, mock_catch, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "ffmpeg"
        )

        input_file = "input.wav"
        output_file = "output.wav"

        remove_silence(input_file, output_file)

        self.assertTrue(mock_catch.called)

        # Check that the exception passed to catch is of the correct type
        args, kwargs = mock_catch.call_args
        self.assertIsInstance(args[0], subprocess.CalledProcessError)


if __name__ == "__main__":
    unittest.main()
