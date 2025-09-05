import unittest
import os
import numpy as np
import soundfile as sf
import tempfile
import shutil
from unittest.mock import patch
from definers import extract_audio_features, catch

class TestExtractAudioFeatures(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.audio_path = os.path.join(self.test_dir, "test_audio.wav")
        self.sample_rate = 22050
        self.duration = 1
        self.audio_data = np.random.randn(self.sample_rate * self.duration)
        sf.write(self.audio_path, self.audio_data, self.sample_rate)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('definers.catch', lambda e: None)
    def test_successful_extraction(self):
        features = extract_audio_features(self.audio_path)
        self.assertIsNotNone(features)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.ndim, 1)
        self.assertGreater(features.shape[0], 20)

    @patch('definers.catch', lambda e: None)
    def test_invalid_audio_path(self):
        features = extract_audio_features("non_existent_audio.wav")
        self.assertIsNone(features)

    @patch('definers.catch', lambda e: None)
    def test_custom_n_mfcc(self):
        n_mfcc = 40
        features = extract_audio_features(self.audio_path, n_mfcc=n_mfcc)
        self.assertIsNotNone(features)
        self.assertGreater(features.shape[0], n_mfcc)

    @patch('librosa.load')
    @patch('definers.catch', lambda e: None)
    def test_internal_librosa_error(self, mock_load):
        mock_load.side_effect = Exception("Simulated Librosa Error")
        features = extract_audio_features(self.audio_path)
        self.assertIsNone(features)

    @patch('definers.catch', lambda e: None)
    def test_corrupt_audio_file(self):
        corrupt_file_path = os.path.join(self.test_dir, "corrupt.wav")
        with open(corrupt_file_path, "w") as f:
            f.write("this is not audio")
        features = extract_audio_features(corrupt_file_path)
        self.assertIsNone(features)

    @patch('definers.catch', lambda e: None)
    def test_output_dtype_is_float32(self):
        features = extract_audio_features(self.audio_path)
        self.assertIsNotNone(features)
        self.assertEqual(features.dtype, np.float32)

if __name__ == '__main__':
    unittest.main()
