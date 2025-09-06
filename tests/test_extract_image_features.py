import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from definers import extract_image_features


class TestExtractImageFeatures(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.image_path = os.path.join(
            self.test_dir, "test_image.png"
        )
        self.width, self.height = 64, 48
        self.image = np.random.randint(
            0, 256, (self.height, self.width, 3), dtype=np.uint8
        )
        cv2.imwrite(self.image_path, self.image)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_successful_extraction(self):
        features = extract_image_features(self.image_path)
        self.assertIsNotNone(features)
        self.assertIsInstance(features, np.ndarray)
        expected_feature_length = (256 * 3) + (
            self.width * self.height
        ) * 2
        self.assertEqual(features.shape[0], expected_feature_length)

    def test_invalid_image_path(self):
        features = extract_image_features("non_existent_image.png")
        self.assertIsNone(features)

    def test_corrupt_image_file(self):
        corrupt_file_path = os.path.join(self.test_dir, "corrupt.png")
        with open(corrupt_file_path, "w") as f:
            f.write("this is not an image")
        features = extract_image_features(corrupt_file_path)
        self.assertIsNone(features)

    @patch("cv2.calcHist")
    def test_internal_cv2_error(self, mock_calchist):
        mock_calchist.side_effect = Exception("Simulated CV2 Error")
        features = extract_image_features(self.image_path)
        self.assertIsNone(features)

    def test_output_dtype_is_float32(self):
        features = extract_image_features(self.image_path)
        self.assertIsNotNone(features)
        self.assertEqual(features.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
