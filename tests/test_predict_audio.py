import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

import definers.audio.features as audio_features_module
from definers.audio.features import predict_audio


class TestPredictAudio(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.audio_path = os.path.join(self.test_dir, "test.wav")
        self.sr = 32000
        self.audio_data = np.random.randn(self.sr * 2)
        sf.write(self.audio_path, self.audio_data, self.sr)
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.random.rand(100)
        self.mock_audio_analysis_backend = MagicMock()
        self.mock_audio_analysis_backend.get_active_audio_timeline.return_value = [
            (0.5, 1.5)
        ]
        self.mock_array_backend = MagicMock()
        self.mock_array_backend.numpy_to_cupy.side_effect = lambda values: (
            values
        )
        self.mock_array_backend.cupy_to_numpy.side_effect = lambda values: (
            values
        )
        self.mock_model_introspection_backend = MagicMock()
        self.mock_model_introspection_backend.is_clusters_model.return_value = (
            False
        )
        self.mock_model_introspection_backend.get_cluster_content.return_value = np.random.rand(
            100
        )
        self.patcher_full_path = patch.object(
            audio_features_module,
            "full_path",
            side_effect=lambda path: path,
        )
        self.patcher_audio_analysis_backend = patch.object(
            audio_features_module,
            "_load_audio_analysis_backend",
            return_value=self.mock_audio_analysis_backend,
        )
        self.patcher_array_backend = patch.object(
            audio_features_module,
            "_load_array_backend",
            return_value=self.mock_array_backend,
        )
        self.patcher_model_introspection_backend = patch.object(
            audio_features_module,
            "_load_model_introspection_backend",
            return_value=self.mock_model_introspection_backend,
        )
        self.patcher_features_to_audio = patch.object(
            audio_features_module,
            "features_to_audio",
            return_value=np.random.randn(self.sr),
        )
        self.patcher_tmp = patch.object(
            audio_features_module,
            "tmp",
            return_value=os.path.join(self.test_dir, "output.wav"),
        )
        self.mock_full_path = self.patcher_full_path.start()
        self.mock_audio_analysis_loader = (
            self.patcher_audio_analysis_backend.start()
        )
        self.mock_array_loader = self.patcher_array_backend.start()
        self.mock_model_introspection_loader = (
            self.patcher_model_introspection_backend.start()
        )
        self.mock_features_to_audio = self.patcher_features_to_audio.start()
        self.mock_tmp = self.patcher_tmp.start()

    def tearDown(self):
        self.patcher_full_path.stop()
        self.patcher_audio_analysis_backend.stop()
        self.patcher_array_backend.stop()
        self.patcher_model_introspection_backend.stop()
        self.patcher_features_to_audio.stop()
        self.patcher_tmp.stop()
        if os.path.exists(self.test_dir):
            import shutil

            shutil.rmtree(self.test_dir)

    def test_successful_prediction_non_cluster_model(self):
        self.mock_model_introspection_backend.is_clusters_model.return_value = (
            False
        )
        result_path = predict_audio(self.mock_model, self.audio_path)
        self.assertIsNotNone(result_path)
        self.assertTrue(os.path.exists(result_path))
        self.mock_audio_analysis_backend.get_active_audio_timeline.assert_called_once_with(
            self.audio_path
        )
        self.mock_model.predict.assert_called()
        self.mock_array_backend.numpy_to_cupy.assert_called()
        self.mock_features_to_audio.assert_called()

    def test_successful_prediction_cluster_model(self):
        self.mock_model_introspection_backend.is_clusters_model.return_value = (
            True
        )
        self.mock_model.predict.return_value = np.array([0])
        result_path = predict_audio(self.mock_model, self.audio_path)
        self.assertIsNotNone(result_path)
        self.assertTrue(os.path.exists(result_path))
        self.mock_model_introspection_backend.get_cluster_content.assert_called_with(
            self.mock_model, 0
        )
        self.mock_array_backend.cupy_to_numpy.assert_called()
        self.mock_features_to_audio.assert_called()

    def test_audio_file_not_found(self):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = predict_audio(self.mock_model, "non_existent_file.wav")

        self.assertEqual(len(w), 0)
        self.assertIsNone(result)

    def test_silent_audio_file(self):
        self.mock_audio_analysis_backend.get_active_audio_timeline.return_value = []
        result_path = predict_audio(self.mock_model, self.audio_path)
        self.assertIsNotNone(result_path)
        self.assertTrue(os.path.exists(result_path))
        self.mock_model.predict.assert_not_called()

    def test_features_to_audio_fails(self):
        self.mock_features_to_audio.return_value = None
        result_path = predict_audio(self.mock_model, self.audio_path)
        self.assertIsNotNone(result_path)
        self.assertTrue(os.path.exists(result_path))
        (output_data, output_sr) = sf.read(result_path)
        self.assertTrue(np.all(output_data == 0))

    def test_model_prediction_raises_exception(self):
        self.mock_model.predict.side_effect = Exception("Prediction failed")
        result = predict_audio(self.mock_model, self.audio_path)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
