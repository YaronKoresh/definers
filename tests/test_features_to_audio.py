import unittest
import numpy as np
from unittest.mock import patch, ANY
from definers import features_to_audio

class TestFeaturesToAudio(unittest.TestCase):

    def setUp(self):
        self.sr = 22050
        self.n_mfcc = 20
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
        self.feature_length = 100 
        self.valid_features = np.random.rand(self.n_mfcc * self.feature_length).astype(np.float32)

    def test_successful_conversion(self):
        audio_waveform = features_to_audio(
            self.valid_features,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        self.assertIsNotNone(audio_waveform)
        self.assertIsInstance(audio_waveform, np.ndarray)
        self.assertTrue(audio_waveform.size > 0)

    def test_padding_logic(self):
        unpadded_features = self.valid_features[:-5] 
        audio_waveform = features_to_audio(
            unpadded_features,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        self.assertIsNotNone(audio_waveform)
        self.assertIsInstance(audio_waveform, np.ndarray)
        self.assertTrue(audio_waveform.size > 0)

    def test_empty_input(self):
        empty_features = np.array([])
        audio_waveform = features_to_audio(
            empty_features,
            sr=self.sr,
            n_mfcc=self.n_mfcc
        )
        self.assertIsNone(audio_waveform)

    def test_zero_frames_after_reshape(self):
        short_features = np.random.rand(self.n_mfcc - 1).astype(np.float32)
        audio_waveform = features_to_audio(
            short_features,
            sr=self.sr,
            n_mfcc=self.n_mfcc
        )
        self.assertIsNone(audio_waveform)

    @patch('definers.librosa.griffinlim')
    def test_griffinlim_failure(self, mock_griffinlim):
        mock_griffinlim.side_effect = Exception("Griffin-Lim failed")
        audio_waveform = features_to_audio(
            self.valid_features,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_mels=self.n_mels,
            n_fft=self.n_fft
        )
        self.assertIsNone(audio_waveform)

    def test_missing_parameters(self):
        audio_waveform = features_to_audio(self.valid_features, sr=None)
        self.assertIsNone(audio_waveform)

    def test_non_finite_values_in_input(self):
        features_with_nan = np.copy(self.valid_features)
        features_with_nan[10] = np.nan
        audio_waveform = features_to_audio(
            features_with_nan,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_mels=self.n_mels,
            n_fft=self.n_fft
        )
        self.assertIsNotNone(audio_waveform)
        self.assertFalse(np.isnan(audio_waveform).any())

if __name__ == '__main__':
    unittest.main()
