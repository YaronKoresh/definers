import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from definers import predict_audio


class TestPredictAudio(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.audio_path = os.path.join(self.test_dir, "test.wav")
        self.sr = 32000
        self.audio_data = np.random.randn(self.sr * 2)
        sf.write(self.audio_path, self.audio_data, self.sr)

        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.random.rand(100)

        self.patcher_timeline = patch(
            "definers.get_active_audio_timeline",
            return_value=[(0.5, 1.5)],
        )
        self.patcher_features_to_audio = patch(
            "definers.features_to_audio",
            return_value=np.random.randn(self.sr),
        )
        self.patcher_tmp = patch(
            "definers.tmp",
            return_value=os.path.join(self.test_dir, "output.wav"),
        )
        self.patcher_is_clusters = patch(
            "definers.is_clusters_model", return_value=False
        )
        self.patcher_get_cluster = patch(
            "definers.get_cluster_content",
            return_value=np.random.rand(100),
        )

        self.mock_timeline = self.patcher_timeline.start()
        self.mock_features_to_audio = (
            self.patcher_features_to_audio.start()
        )
        self.mock_tmp = self.patcher_tmp.start()
        self.mock_is_clusters = self.patcher_is_clusters.start()
        self.mock_get_cluster = self.patcher_get_cluster.start()

    def tearDown(self):
        self.patcher_timeline.stop()
        self.patcher_features_to_audio.stop()
        self.patcher_tmp.stop()
        self.patcher_is_clusters.stop()
        self.patcher_get_cluster.stop()
        if os.path.exists(self.test_dir):
            import shutil

            shutil.rmtree(self.test_dir)

    def test_successful_prediction_non_cluster_model(self):
        self.mock_is_clusters.return_value = False
        result_path = predict_audio(self.mock_model, self.audio_path)

        self.assertIsNotNone(result_path)
        self.assertTrue(os.path.exists(result_path))
        self.mock_timeline.assert_called_once_with(self.audio_path)
        self.mock_model.predict.assert_called()
        self.mock_features_to_audio.assert_called()

    def test_successful_prediction_cluster_model(self):
        self.mock_is_clusters.return_value = True
        self.mock_model.predict.return_value = np.array([0])
        result_path = predict_audio(self.mock_model, self.audio_path)

        self.assertIsNotNone(result_path)
        self.assertTrue(os.path.exists(result_path))
        self.mock_get_cluster.assert_called_with(self.mock_model, 0)
        self.mock_features_to_audio.assert_called()

    def test_audio_file_not_found(self):
        result = predict_audio(
            self.mock_model, "non_existent_file.wav"
        )
        self.assertIsNone(result)

    def test_silent_audio_file(self):
        self.mock_timeline.return_value = []
        result_path = predict_audio(self.mock_model, self.audio_path)

        self.assertIsNotNone(result_path)
        self.assertTrue(os.path.exists(result_path))
        self.mock_model.predict.assert_not_called()

    def test_features_to_audio_fails(self):
        self.mock_features_to_audio.return_value = None
        result_path = predict_audio(self.mock_model, self.audio_path)

        self.assertIsNotNone(result_path)
        self.assertTrue(os.path.exists(result_path))

        output_data, output_sr = sf.read(result_path)
        self.assertTrue(np.all(output_data == 0))

    def test_model_prediction_raises_exception(self):
        self.mock_model.predict.side_effect = Exception(
            "Prediction failed"
        )
        result = predict_audio(self.mock_model, self.audio_path)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
