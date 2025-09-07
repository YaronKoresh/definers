import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from definers import extract_audio_features


class TestExtractAudioFeatures(unittest.TestCase):
    def setUp(self):
        self.audio_path = "test_audio.wav"
        self.sample_rate = 22050
        self.dummy_audio_data = np.random.randn(self.sample_rate * 2)

    def test_successful_extraction(self):
        mock_librosa = MagicMock()
        mock_librosa.load.return_value = (
            self.dummy_audio_data,
            self.sample_rate,
        )

        with patch.dict("sys.modules", {"librosa": mock_librosa}):
            features = extract_audio_features(self.audio_path)

        self.assertIsNotNone(features)
        self.assertIsInstance(features, np.ndarray)

    def test_custom_n_mfcc(self):
        n_mfcc = 40
        mock_librosa = MagicMock()
        mock_librosa.load.return_value = (
            self.dummy_audio_data,
            self.sample_rate,
        )
        mock_librosa.feature.mfcc.return_value = np.random.rand(
            n_mfcc, 10
        )

        with patch.dict("sys.modules", {"librosa": mock_librosa}):
            features = extract_audio_features(
                self.audio_path, n_mfcc=n_mfcc
            )

        self.assertIsNotNone(features)
        mock_librosa.feature.mfcc.assert_called_with(
            y=self.dummy_audio_data,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=2048,
            n_mels=80,
        )

    def test_audio_loading_error(self):
        mock_librosa = MagicMock()
        mock_librosa.load.side_effect = Exception("File not found")

        with patch.dict("sys.modules", {"librosa": mock_librosa}):
            features = extract_audio_features(self.audio_path)

        self.assertIsNone(features)

    def test_output_dtype_is_float32(self):
        mock_librosa = MagicMock()
        mock_librosa.load.return_value = (
            self.dummy_audio_data,
            self.sample_rate,
        )

        with patch.dict("sys.modules", {"librosa": mock_librosa}):
            features = extract_audio_features(self.audio_path)

        self.assertIsNotNone(features)
        self.assertEqual(features.dtype, np.float32)

    def test_feature_extraction_error(self):
        mock_librosa = MagicMock()
        mock_librosa.load.return_value = (
            self.dummy_audio_data,
            self.sample_rate,
        )
        mock_librosa.feature.mfcc.side_effect = Exception(
            "Feature error"
        )

        with patch.dict("sys.modules", {"librosa": mock_librosa}):
            features = extract_audio_features(self.audio_path)

        self.assertIsNone(features)


if __name__ == "__main__":
    unittest.main()
