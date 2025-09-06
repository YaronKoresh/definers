import unittest
from unittest.mock import patch

import cv2
import numpy as np

from definers import features_to_image


class TestFeaturesToImage(unittest.TestCase):

    def setUp(self):
        self.height = 100
        self.width = 150
        self.channels = 3
        self.hist_size = 256 * self.channels
        self.lbp_size = self.height * self.width
        self.edge_size = self.height * self.width
        self.total_feature_length = (
            self.hist_size + self.lbp_size + self.edge_size
        )
        self.valid_features = np.random.rand(
            self.total_feature_length
        ).astype(np.float32)

    @patch("definers.cv2.normalize")
    @patch("definers.cv2.addWeighted")
    @patch("definers.cv2.cvtColor")
    def test_successful_reconstruction(
        self, mock_cvtColor, mock_addWeighted, mock_normalize
    ):

        mock_normalize.side_effect = lambda src, dst, alpha, beta, norm_type, dtype: np.zeros_like(
            src, dtype=np.uint8
        )
        mock_addWeighted.side_effect = (
            lambda src1, alpha, src2, beta, gamma: np.zeros_like(
                src1, dtype=np.uint8
            )
        )
        mock_cvtColor.side_effect = lambda src, code: np.zeros(
            (
                (self.height, self.width, self.channels)
                if code == cv2.COLOR_GRAY2BGR
                else (self.height, self.width)
            ),
            dtype=np.uint8,
        )

        with patch(
            "definers.image_shape",
            (self.height, self.width, self.channels),
        ):
            reconstructed_image = features_to_image(
                self.valid_features
            )

        self.assertIsNotNone(reconstructed_image)
        self.assertIsInstance(reconstructed_image, np.ndarray)
        self.assertEqual(
            reconstructed_image.shape,
            (self.height, self.width, self.channels),
        )
        self.assertEqual(reconstructed_image.dtype, np.uint8)

    def test_incorrect_feature_length(self):
        invalid_features = np.random.rand(
            self.total_feature_length - 10
        ).astype(np.float32)
        with patch(
            "definers.image_shape",
            (self.height, self.width, self.channels),
        ):
            reconstructed_image = features_to_image(invalid_features)
        self.assertIsNone(reconstructed_image)

    def test_zero_features_input(self):
        zero_features = np.zeros(
            self.total_feature_length, dtype=np.float32
        )
        with patch(
            "definers.image_shape",
            (self.height, self.width, self.channels),
        ):
            reconstructed_image = features_to_image(zero_features)

        self.assertIsNotNone(reconstructed_image)
        self.assertTrue(np.all(reconstructed_image == 0))

    def test_high_value_features(self):
        high_value_features = (
            np.ones(self.total_feature_length, dtype=np.float32)
            * 1000
        )
        with patch(
            "definers.image_shape",
            (self.height, self.width, self.channels),
        ):
            reconstructed_image = features_to_image(
                high_value_features
            )
        self.assertIsNotNone(reconstructed_image)

    @patch("definers.np.zeros")
    def test_exception_handling(self, mock_zeros):
        mock_zeros.side_effect = Exception("Test exception")
        with patch(
            "definers.image_shape",
            (self.height, self.width, self.channels),
        ):
            reconstructed_image = features_to_image(
                self.valid_features
            )
        self.assertIsNone(reconstructed_image)


if __name__ == "__main__":
    unittest.main()
