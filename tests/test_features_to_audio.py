import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
AUDIO_FEATURES_MODULE = _load_module(
    "_test_features_to_audio_module",
    ROOT / "src" / "definers" / "audio" / "features.py",
)
features_to_audio = AUDIO_FEATURES_MODULE.features_to_audio


class TestFeaturesToAudio(unittest.TestCase):
    def setUp(self):
        self.valid_features = np.random.rand(20 * 100).astype(np.float32)
        self.mock_librosa = MagicMock()
        self.mock_librosa.feature.inverse.mfcc_to_mel.return_value = (
            np.random.rand(80, 100)
        )
        self.mock_librosa.db_to_amplitude.return_value = np.random.rand(80, 100)
        self.mock_librosa.feature.inverse.mel_to_stft.return_value = (
            np.random.rand(1025, 100)
        )
        self.mock_librosa.griffinlim.return_value = np.random.rand(22050)

    def test_successful_conversion(self):
        with patch.dict("sys.modules", {"librosa": self.mock_librosa}):
            audio_waveform = features_to_audio(self.valid_features)
        self.assertIsNotNone(audio_waveform)
        self.assertIsInstance(audio_waveform, np.ndarray)

    def test_padding_logic(self):
        unpadded_features = np.random.rand(20 * 100 + 5).astype(np.float32)
        with patch.dict("sys.modules", {"librosa": self.mock_librosa}):
            features_to_audio(unpadded_features, n_mfcc=20)
        self.mock_librosa.feature.inverse.mfcc_to_mel.assert_called()

    def test_uses_default_parameters(self):
        with patch.dict("sys.modules", {"librosa": self.mock_librosa}):
            features_to_audio(self.valid_features)
        self.mock_librosa.feature.inverse.mel_to_stft.assert_called_with(
            M=unittest.mock.ANY, sr=32000, n_fft=2048
        )
        self.mock_librosa.griffinlim.assert_called_with(
            unittest.mock.ANY,
            n_fft=2048,
            hop_length=512,
            n_iter=unittest.mock.ANY,
        )

    def test_griffinlim_failure(self):
        self.mock_librosa.griffinlim.side_effect = Exception(
            "Griffin-Lim failed"
        )
        with patch.dict("sys.modules", {"librosa": self.mock_librosa}):
            audio_waveform = features_to_audio(self.valid_features)
        self.assertIsNone(audio_waveform)

    def test_empty_features_input(self):
        with patch.dict("sys.modules", {"librosa": self.mock_librosa}):
            audio_waveform = features_to_audio(np.array([]))
        self.assertIsNone(audio_waveform)


if __name__ == "__main__":
    unittest.main()
