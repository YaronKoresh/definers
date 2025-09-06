import subprocess
import unittest
from unittest.mock import patch

from definers import compact_audio


class TestCompactAudio(unittest.TestCase):

    @patch("subprocess.run")
    def test_compact_audio_success(self, mock_run):
        input_file = "input.mp3"
        output_file = "output.mp3"

        result = compact_audio(input_file, output_file)

        self.assertEqual(result, output_file)
        expected_command = [
            "ffmpeg",
            "-y",
            "-i",
            input_file,
            "-ar",
            "16000",
            "-ab",
            "320k",
            "-ac",
            "1",
            output_file,
        ]
        mock_run.assert_called_once_with(expected_command, check=True)

    @patch("subprocess.run")
    @patch("definers.catch")
    def test_compact_audio_failure(self, mock_catch, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "ffmpeg"
        )

        input_file = "input.mp3"
        output_file = "output.mp3"

        result = compact_audio(input_file, output_file)

        # The function should return None on failure, but the test needs to check that catch was called
        # and the result is not the output file.
        self.assertIsNone(result)
        mock_catch.assert_called_once()
        self.assertIsInstance(
            mock_catch.call_args[0][0], subprocess.CalledProcessError
        )


if __name__ == "__main__":
    unittest.main()
