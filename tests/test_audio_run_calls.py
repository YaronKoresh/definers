import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = ROOT / "src" / "definers" / "audio"
AUDIO_UTILS_MODULE = _load_module("_test_audio_utils", AUDIO_ROOT / "utils.py")
TEST_AUDIO_PACKAGE = "_test_audio_package"
AUDIO_PACKAGE = types.ModuleType(TEST_AUDIO_PACKAGE)
AUDIO_PACKAGE.__path__ = [str(AUDIO_ROOT)]
sys.modules[TEST_AUDIO_PACKAGE] = AUDIO_PACKAGE
sys.modules[f"{TEST_AUDIO_PACKAGE}.analysis"] = types.SimpleNamespace(
    get_active_audio_timeline=MagicMock()
)
_load_module(f"{TEST_AUDIO_PACKAGE}.io", AUDIO_ROOT / "io.py")
AUDIO_PREVIEW_MODULE = _load_module(
    f"{TEST_AUDIO_PACKAGE}.preview", AUDIO_ROOT / "preview.py"
)
get_audio_duration = AUDIO_PREVIEW_MODULE.get_audio_duration
stretch_audio = AUDIO_UTILS_MODULE.stretch_audio


class TestAudioRunCalls(unittest.TestCase):
    @patch.object(
        AUDIO_UTILS_MODULE, "normalize_audio_to_peak", side_effect=lambda p: p
    )
    @patch.object(AUDIO_UTILS_MODULE, "run")
    def test_stretch_audio_uses_list(self, mock_run, mock_normalize):
        import tempfile

        inp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        inp.close()
        out = inp.name.replace(".wav", "_out.wav")
        stretch_audio(inp.name, output_path=out, speed_factor=1.1)
        self.assertTrue(mock_run.called)
        args = mock_run.call_args[0][0]
        self.assertIsInstance(args, list)
        self.assertEqual(args[0], "rubberband")

    @patch("definers.logger.logger.exception")
    def test_get_audio_duration_logs_error(self, mock_logger_exc):
        res = get_audio_duration("no_such_file.wav")
        self.assertIsNone(res)
        mock_logger_exc.assert_called_once()


if __name__ == "__main__":
    unittest.main()
