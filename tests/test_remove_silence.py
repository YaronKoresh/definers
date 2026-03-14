import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
AUDIO_IO_MODULE = _load_module(
    "_test_remove_silence_io", ROOT / "src" / "definers" / "audio" / "io.py"
)
remove_silence = AUDIO_IO_MODULE.remove_silence


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
    @patch("definers.file_ops.catch")
    def test_remove_silence_failure(self, mock_catch, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")
        input_file = "input.wav"
        output_file = "output.wav"
        remove_silence(input_file, output_file)
        self.assertTrue(mock_catch.called)
        (args, kwargs) = mock_catch.call_args
        self.assertIsInstance(args[0], subprocess.CalledProcessError)


if __name__ == "__main__":
    unittest.main()
