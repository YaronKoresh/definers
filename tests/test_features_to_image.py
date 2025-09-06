import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from definers import features_to_image

class TestFeaturesToImage(unittest.TestCase):
    def setUp(self):
        self.height = 64
        self.width = 64
        self.channels = 3
        self.hist_size = 256 * self.channels
        self.lbp_size = self.height * self.width
        self.edge_size = self.height * self.width
        self.total_features = self.hist_size + self.lbp_size + self.edge_size
        self.features = np.random.rand(self.total_features).astype(np.float32)

        self.mock_cv2 = MagicMock()
        self.mock_cv2.normalize.side_effect = lambda src, dst, alpha, beta, norm_type, dtype: (src * 255).astype(np.uint8)
        self.mock_cv2.addWeighted.side_effect = lambda src1, alpha, src2, beta, gamma: src1
        self.mock_cv2.cvtColor.side_effect = lambda src, code: src

    def test_successful_reconstruction(self):
        with patch.dict('sys.modules', {'cv2': self.mock_cv2}):
            image = features_to_image(
                self.features, image_shape=(self.height, self.width, self.channels)
            )
        self.assertIsNotNone(image)
        self.assertIsInstance(image, np.ndarray)

    def test_correct_output_shape(self):
        with patch.dict('sys.modules', {'cv2': self.mock_cv2}):
            image = features_to_image(
                self.features, image_shape=(self.height, self.width, self.channels)
            )
        self.assertEqual(image.shape, (self.height, self.width, self.channels))

    def test_correct_output_dtype(self):
        with patch.dict('sys.modules', {'cv2': self.mock_cv2}):
            image = features_to_image(
                self.features, image_shape=(self.height, self.width, self.channels)
            )
        self.assertEqual(image.dtype, np.uint8)

    def test_invalid_feature_size(self):
        invalid_features = np.random.rand(self.total_features - 10).astype(np.float32)
        with patch.dict('sys.modules', {'cv2': self.mock_cv2}):
            image = features_to_image(
                invalid_features, image_shape=(self.height, self.width, self.channels)
            )
        self.assertIsNone(image)

    def test_exception_handling(self):
        self.mock_cv2.normalize.side_effect = Exception("Test exception")
        with patch.dict('sys.modules', {'cv2': self.mock_cv2}):
            image = features_to_image(
                self.features, image_shape=(self.height, self.width, self.channels)
            )
        self.assertIsNone(image)

if __name__ == "__main__":
    unittest.main()

