import importlib
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import definers.os_utils as os_utils
import definers.path_utils as path_utils

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

from definers.media.video_helpers import features_to_video


class TestFeaturesToVideo(unittest.TestCase):
    def setUp(self):
        self.height = 32
        self.width = 32
        self.channels = 3
        self.video_shape = (self.height, self.width, self.channels)
        self.hist_size = 256 * self.channels
        self.lbp_size = self.height * self.width
        self.edge_size = self.height * self.width
        self.total_features_per_frame = (
            self.hist_size + self.lbp_size + self.edge_size
        )
        self.num_frames = 2
        self.features = np.arange(
            self.num_frames * self.total_features_per_frame,
            dtype=np.float32,
        ).reshape(self.num_frames, self.total_features_per_frame)
        self.mock_cv2 = MagicMock()
        self.mock_cv2.VideoWriter_fourcc.return_value = "mp4v"
        self.mock_writer = MagicMock()
        self.mock_cv2.VideoWriter.return_value = self.mock_writer
        self.mock_cv2.normalize.side_effect = (
            lambda src, dst, alpha, beta, norm_type, dtype: (src * 255).astype(
                np.uint8
            )
        )
        self.mock_cv2.addWeighted.side_effect = (
            lambda src1, alpha, src2, beta, gamma: src1
        )
        self.mock_cv2.cvtColor.side_effect = lambda src, code: src

    def get_runtime_package(self):
        runtime_package = importlib.import_module("definers")
        self.assertIn("definers", sys.modules)
        self.assertIs(sys.modules["definers"], runtime_package)
        return runtime_package

    def test_package_identity_for_runtime_import(self):
        runtime_package = self.get_runtime_package()

        self.assertIs(importlib.import_module("definers"), runtime_package)

    def test_successful_video_generation(self):
        runtime_package = self.get_runtime_package()

        with (
            patch.object(
                runtime_package,
                "tmp",
                create=True,
                return_value="/fake/video.mp4",
            ) as mock_tmp,
            patch.dict("sys.modules", {"cv2": self.mock_cv2}),
        ):
            self.assertIs(importlib.import_module("definers"), runtime_package)
            self.assertIs(importlib.import_module("definers").tmp, mock_tmp)
            result = features_to_video(
                self.features, video_shape=self.video_shape
            )
        self.assertEqual(result, "/fake/video.mp4")
        self.mock_cv2.VideoWriter.assert_called_with(
            "/fake/video.mp4", "mp4v", 24, (self.width, self.height)
        )
        self.assertEqual(self.mock_writer.write.call_count, self.num_frames)
        self.mock_writer.release.assert_called_once()
        mock_tmp.assert_called_once_with("mp4")

    def test_patched_tmp_visible_from_fresh_runtime_import(self):
        runtime_package = self.get_runtime_package()

        with (
            patch.object(
                runtime_package,
                "tmp",
                create=True,
                return_value="/fresh/video.mp4",
            ) as mock_tmp,
            patch.dict("sys.modules", {"cv2": self.mock_cv2}),
        ):
            fresh_package = importlib.import_module("definers")
            self.assertIs(fresh_package, runtime_package)
            self.assertIs(fresh_package.tmp, mock_tmp)
            result = features_to_video(
                self.features[:1], video_shape=self.video_shape
            )
        self.assertEqual(result, "/fresh/video.mp4")
        mock_tmp.assert_called_once_with("mp4")

    def test_exception_during_processing(self):
        runtime_package = self.get_runtime_package()
        self.mock_cv2.normalize.side_effect = Exception(
            "Test processing exception"
        )
        with (
            patch.object(
                runtime_package,
                "tmp",
                create=True,
                return_value="/fake/video.mp4",
            ) as mock_tmp,
            patch.dict("sys.modules", {"cv2": self.mock_cv2}),
        ):
            self.assertIs(importlib.import_module("definers"), runtime_package)
            self.assertIs(importlib.import_module("definers").tmp, mock_tmp)
            result = features_to_video(
                self.features, video_shape=self.video_shape
            )
        self.assertFalse(result)
        mock_tmp.assert_called_once_with("mp4")

    def test_empty_features_input(self):
        empty_features = np.array([])
        result = features_to_video(empty_features)
        self.assertFalse(result)

    def test_none_features_input(self):
        result = features_to_video(None)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
