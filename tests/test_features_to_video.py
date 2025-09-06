import os
import unittest
from unittest.mock import ANY, MagicMock, patch

import numpy as np

from definers import features_to_video


class TestFeaturesToVideo(unittest.TestCase):

    def setUp(self):
        self.height = 64
        self.width = 64
        self.channels = 3
        self.hist_size = 256 * self.channels
        self.lbp_size = self.height * self.width
        self.edge_size = self.height * self.width
        self.feature_length_per_frame = (
            self.hist_size + self.lbp_size + self.edge_size
        )
        self.num_frames = 5
        self.valid_features = np.random.rand(
            self.num_frames, self.feature_length_per_frame
        ).astype(np.float32)
        self.output_path = "test_video.mp4"

    def tearDown(self):
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

    @patch("definers.tmp", return_value="test_video.mp4")
    @patch("definers.cv2.VideoWriter")
    @patch("definers.video_shape", (64, 64, 3))
    def test_successful_video_generation(
        self, mock_video_writer, mock_tmp
    ):
        mock_writer_instance = MagicMock()
        mock_video_writer.return_value = mock_writer_instance

        result = features_to_video(self.valid_features, fps=24)

        self.assertEqual(result, self.output_path)
        mock_video_writer.assert_called_with(
            self.output_path, ANY, 24, (self.width, self.height)
        )
        self.assertEqual(
            mock_writer_instance.write.call_count, self.num_frames
        )
        mock_writer_instance.release.assert_called_once()

    @patch("definers.tmp", return_value="test_video.mp4")
    @patch("definers.cv2.VideoWriter")
    @patch("definers.video_shape", (64, 64, 3))
    def test_exception_during_processing(
        self, mock_video_writer, mock_tmp
    ):
        mock_writer_instance = MagicMock()
        mock_writer_instance.write.side_effect = Exception(
            "Failed to write frame"
        )
        mock_video_writer.return_value = mock_writer_instance

        result = features_to_video(self.valid_features)

        self.assertFalse(result)

    def test_empty_features_input(self):
        empty_features = np.array([])
        result = features_to_video(empty_features)
        self.assertFalse(result)

    @patch("definers.video_shape", (64, 64, 3))
    def test_invalid_feature_shape(self):
        invalid_features = np.random.rand(
            self.num_frames, self.feature_length_per_frame - 10
        )
        result = features_to_video(invalid_features)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
