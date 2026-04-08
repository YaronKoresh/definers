import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import definers.os_utils as os_utils
import definers.path_utils as path_utils
from tests.optional_dependency_stubs import build_fake_cv2_module

if not hasattr(os_utils, "get_python_version"):
    os_utils.get_python_version = lambda: "3.10"
if not hasattr(os_utils, "get_linux_distribution"):
    os_utils.get_linux_distribution = lambda: "linux"

for _name, _value in {
    "normalize_path": lambda path: str(path),
    "full_path": lambda *parts: "/".join(
        str(part) for part in parts if str(part)
    ),
    "paths": lambda *patterns: [],
    "unique": lambda items: list(dict.fromkeys(items)),
    "cwd": lambda: ".",
    "parent_directory": lambda path: "",
    "path_end": lambda path: str(path).rsplit("/", 1)[-1].rsplit("\\", 1)[-1],
    "path_ext": lambda path: (
        "" if "." not in str(path) else "." + str(path).rsplit(".", 1)[-1]
    ),
    "path_name": lambda path: (
        str(path).rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0]
    ),
    "tmp": lambda *args, **kwargs: "/tmp/mock",
    "secure_path": lambda path, *args, **kwargs: path,
}.items():
    if not hasattr(path_utils, _name):
        setattr(path_utils, _name, _value)

from definers.image import extract_image_features


class TestExtractImageFeatures(unittest.TestCase):
    def setUp(self):
        self.cv2_module = build_fake_cv2_module()
        self.cv2_patcher = patch.dict("sys.modules", {"cv2": self.cv2_module})
        self.cv2_patcher.start()
        self.test_dir = tempfile.mkdtemp()
        self.image_path = os.path.join(self.test_dir, "test_image.png")
        (self.width, self.height) = (64, 48)
        self.image = np.random.randint(
            0, 256, (self.height, self.width, 3), dtype=np.uint8
        )
        self.cv2_module.imwrite(self.image_path, self.image)

    def tearDown(self):
        self.cv2_patcher.stop()
        shutil.rmtree(self.test_dir)

    def test_successful_extraction(self):
        features = extract_image_features(self.image_path)
        self.assertIsNotNone(features)
        self.assertIsInstance(features, np.ndarray)
        expected_feature_length = 256 * 3 + self.width * self.height * 2
        self.assertEqual(features.shape[0], expected_feature_length)

    def test_invalid_image_path(self):
        features = extract_image_features("non_existent_image.png")
        self.assertIsNone(features)

    def test_corrupt_image_file(self):
        corrupt_file_path = os.path.join(self.test_dir, "corrupt.png")
        self.cv2_module._image_store[corrupt_file_path] = None
        with open(corrupt_file_path, "w") as f:
            f.write("this is not an image")
        features = extract_image_features(corrupt_file_path)
        self.assertIsNone(features)

    @patch("cv2.calcHist")
    @patch("definers.logger.exception", create=True)
    def test_internal_cv2_error(self, mock_logger_exc, mock_calchist):
        mock_calchist.side_effect = Exception("Simulated CV2 Error")
        features = extract_image_features(self.image_path)
        self.assertIsNone(features)
        mock_logger_exc.assert_called_once()

    def test_output_dtype_is_float32(self):
        features = extract_image_features(self.image_path)
        self.assertIsNotNone(features)
        self.assertEqual(features.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
