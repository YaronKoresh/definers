import os
import unittest
from unittest.mock import MagicMock, call, mock_open, patch

import numpy as np

from definers import predict


class TestPredict(unittest.TestCase):

    def setUp(self):
        self.model_path = "test_model.joblib"
        self.prediction_file_txt = "test_input.txt"
        self.prediction_file_audio = "test_input.wav"
        self.prediction_file_image = "test_input.png"
        self.mock_model = MagicMock()

    @patch("definers.joblib.load")
    @patch("definers.read")
    @patch("definers.create_vectorizer")
    @patch("definers.extract_text_features")
    @patch("definers.numpy_to_cupy")
    @patch("definers.one_dim_numpy")
    @patch("definers.is_clusters_model", return_value=False)
    @patch("definers.guess_numpy_type", return_value="text")
    @patch("definers.features_to_text")
    @patch("builtins.open", new_callable=mock_open)
    @patch("definers.random_string", return_value="random")
    def test_predict_text_success(
        self,
        mock_random_string,
        mock_file_open,
        mock_features_to_text,
        mock_guess_type,
        mock_is_clusters,
        mock_one_dim,
        mock_to_cupy,
        mock_extract,
        mock_create_vec,
        mock_read,
        mock_joblib_load,
    ):

        mock_joblib_load.return_value = self.mock_model
        mock_read.return_value = "This is a test text."
        mock_vectorizer_instance = MagicMock()
        mock_create_vec.return_value = mock_vectorizer_instance
        mock_extracted_features = np.array([0.1, 0.2, 0.3])
        mock_extract.return_value = mock_extracted_features
        mock_to_cupy.side_effect = lambda x: x
        mock_one_dim.return_value = mock_extracted_features.flatten()

        mock_prediction_result = np.array([0.4, 0.5, 0.6])
        self.mock_model.predict.return_value = mock_prediction_result

        mock_features_to_text.return_value = "predicted text"

        result_path = predict(
            self.prediction_file_txt, self.model_path
        )

        self.assertEqual(result_path, "random.txt")
        mock_joblib_load.assert_called_with(self.model_path)
        mock_read.assert_called_with(self.prediction_file_txt)
        self.mock_model.predict.assert_called_with(
            mock_extracted_features.flatten()
        )
        mock_features_to_text.assert_called_once()
        mock_file_open.assert_called_with("random.txt", "w")
        mock_file_open().write.assert_called_with("predicted text")

    @patch("definers.joblib.load")
    @patch("definers.predict_audio")
    def test_predict_audio_direct_call(
        self, mock_predict_audio, mock_joblib_load
    ):
        mock_joblib_load.return_value = self.mock_model
        mock_predict_audio.return_value = "predicted_audio.wav"

        result_path = predict(
            self.prediction_file_audio, self.model_path
        )

        self.assertEqual(result_path, "predicted_audio.wav")
        mock_joblib_load.assert_called_with(self.model_path)
        mock_predict_audio.assert_called_with(
            self.mock_model, self.prediction_file_audio
        )

    @patch("definers.joblib.load")
    @patch("definers.load_as_numpy")
    @patch("definers.numpy_to_cupy")
    @patch("definers.one_dim_numpy")
    @patch("definers.is_clusters_model", return_value=False)
    @patch("definers.guess_numpy_type", return_value="image")
    @patch("definers.features_to_image")
    @patch("definers.iio.imwrite")
    @patch("definers.cupy_to_numpy")
    @patch("definers.random_string", return_value="random_img")
    def test_predict_image_success(
        self,
        mock_random_string,
        mock_cupy_to_numpy,
        mock_imwrite,
        mock_features_to_image,
        mock_guess_type,
        mock_is_clusters,
        mock_one_dim,
        mock_to_cupy,
        mock_load_numpy,
        mock_joblib_load,
    ):
        mock_joblib_load.return_value = self.mock_model
        mock_input_data = np.random.rand(100, 100, 3)
        mock_load_numpy.return_value = mock_input_data
        mock_to_cupy.side_effect = lambda x: x
        mock_one_dim.return_value = mock_input_data.flatten()

        mock_prediction_result = np.random.rand(100 * 100 * 3)
        self.mock_model.predict.return_value = mock_prediction_result

        mock_reconstructed_image = np.zeros(
            (50, 50, 3), dtype=np.uint8
        )
        mock_features_to_image.return_value = mock_reconstructed_image
        mock_cupy_to_numpy.side_effect = lambda x: (
            x if isinstance(x, np.ndarray) else np.array(x)
        )

        result_path = predict(
            self.prediction_file_image, self.model_path
        )

        self.assertEqual(result_path, "random_img.png")
        mock_load_numpy.assert_called_with(self.prediction_file_image)
        self.mock_model.predict.assert_called_with(
            mock_input_data.flatten()
        )
        mock_features_to_image.assert_called_once()
        mock_imwrite.assert_called_once()

    @patch("definers.joblib.load", return_value=None)
    def test_predict_model_load_fail(self, mock_joblib_load):
        result = predict(self.prediction_file_txt, self.model_path)
        self.assertIsNone(result)

    @patch("definers.joblib.load")
    @patch("definers.load_as_numpy", return_value=None)
    def test_predict_input_load_fail(
        self, mock_load_numpy, mock_joblib_load
    ):
        mock_joblib_load.return_value = self.mock_model
        result = predict("some_other_file.data", self.model_path)
        self.assertIsNone(result)

    @patch("definers.joblib.load")
    @patch("definers.load_as_numpy")
    @patch("definers.numpy_to_cupy")
    @patch("definers.one_dim_numpy")
    def test_predict_prediction_fail(
        self,
        mock_one_dim,
        mock_to_cupy,
        mock_load_numpy,
        mock_joblib_load,
    ):
        mock_joblib_load.return_value = self.mock_model
        mock_load_numpy.return_value = np.array([1, 2, 3])
        mock_to_cupy.side_effect = lambda x: x
        mock_one_dim.return_value = np.array([1, 2, 3])
        self.mock_model.predict.return_value = None

        result = predict("some_file.data", self.model_path)
        self.assertIsNone(result)

    @patch("definers.joblib.load")
    @patch("definers.load_as_numpy")
    @patch("definers.numpy_to_cupy")
    @patch("definers.one_dim_numpy")
    @patch("definers.is_clusters_model", return_value=True)
    @patch("definers.get_cluster_content")
    @patch("definers.guess_numpy_type", return_value="text")
    @patch("definers.features_to_text")
    @patch("builtins.open", new_callable=mock_open)
    def test_predict_with_clustering(
        self,
        mock_file_open,
        mock_features_to_text,
        mock_guess_type,
        mock_get_cluster,
        mock_is_clusters,
        mock_one_dim,
        mock_to_cupy,
        mock_load_numpy,
        mock_joblib_load,
    ):
        mock_joblib_load.return_value = self.mock_model
        mock_load_numpy.return_value = np.array([1, 2, 3])
        mock_to_cupy.side_effect = lambda x: x
        mock_one_dim.return_value = np.array([1, 2, 3])

        self.mock_model.predict.return_value = np.array([0])
        mock_cluster_content = np.array([0.7, 0.8, 0.9])
        mock_get_cluster.return_value = mock_cluster_content
        mock_features_to_text.return_value = "clustered text"

        predict("some_file.data", self.model_path)

        mock_get_cluster.assert_called_with(self.mock_model, 0)
        mock_features_to_text.assert_called_with(mock_cluster_content)
        mock_file_open().write.assert_called_with("clustered text")


if __name__ == "__main__":
    unittest.main()
