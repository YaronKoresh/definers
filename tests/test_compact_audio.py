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
    "_test_compact_audio_io", ROOT / "src" / "definers" / "audio" / "io.py"
)
compact_audio = AUDIO_IO_MODULE.compact_audio


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
            "32000",
            "-ab",
            "96k",
            "-ac",
            "1",
            output_file,
        ]
        mock_run.assert_called_once_with(expected_command, check=True)

    @patch("subprocess.run")
    @patch("definers.file_ops.catch")
    def test_compact_audio_failure(self, mock_catch, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")
        input_file = "input.mp3"
        output_file = "output.mp3"
        result = compact_audio(input_file, output_file)
        self.assertIsNone(result)
        mock_catch.assert_called_once()
        self.assertIsInstance(
            mock_catch.call_args[0][0], subprocess.CalledProcessError
        )


if __name__ == "__main__":
    unittest.main()
