import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from definers import extract_video_features


class TestExtractVideoFeatures(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.video_path = os.path.join(
            self.test_dir, "test_video.mp4"
        )
        self.width, self.height = 64, 48
        self.frame_count = 30
        self.fps = 10
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            self.video_path,
            fourcc,
            self.fps,
            (self.width, self.height),
        )
        for _ in range(self.frame_count):
            frame = np.random.randint(
                0, 256, (self.height, self.width, 3), dtype=np.uint8
            )
            out.write(frame)
        out.release()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("definers.catch", lambda e: None)
    def test_successful_extraction(self):
        features = extract_video_features(self.video_path)
        self.assertIsNotNone(features)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[0], 3)
        expected_feature_length = (256 * 3) + (
            self.width * self.height
        ) * 2
        self.assertEqual(features.shape[1], expected_feature_length)

    @patch("definers.catch", lambda e: None)
    def test_invalid_video_path(self):
        features = extract_video_features("non_existent_video.mp4")
        self.assertIsNone(features)

    @patch("definers.catch", lambda e: None)
    def test_custom_frame_interval(self):
        frame_interval = 5
        features = extract_video_features(
            self.video_path, frame_interval=frame_interval
        )
        self.assertIsNotNone(features)
        expected_frames = self.frame_count // frame_interval
        self.assertEqual(features.shape[0], expected_frames)

    @patch("definers.catch", lambda e: None)
    def test_empty_video(self):
        empty_video_path = os.path.join(self.test_dir, "empty.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            empty_video_path,
            fourcc,
            self.fps,
            (self.width, self.height),
        )
        out.release()
        features = extract_video_features(empty_video_path)
        self.assertIsNone(features)

    @patch("definers.catch", lambda e: None)
    def test_interval_larger_than_frames(self):
        frame_interval = 40
        features = extract_video_features(
            self.video_path, frame_interval=frame_interval
        )
        self.assertIsNotNone(features)
        self.assertEqual(features.shape[0], 1)

    @patch("cv2.cvtColor")
    @patch("definers.catch", lambda e: None)
    def test_internal_cv2_error(self, mock_cvtcolor):
        mock_cvtcolor.side_effect = Exception("Simulated CV2 Error")
        features = extract_video_features(self.video_path)
        self.assertIsNone(features)


if __name__ == "__main__":
    unittest.main()
