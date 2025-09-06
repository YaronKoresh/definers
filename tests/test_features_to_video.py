import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from definers import features_to_video, tmp

class TestFeaturesToVideo(unittest.TestCase):
    def setUp(self):
        self.height = 32
        self.width = 32
        self.channels = 3
        self.video_shape = (self.height, self.width, self.channels)
        self.hist_size = 256 * self.channels
        self.lbp_size = self.height * self.width
        self.edge_size = self.height * self.width
        self.total_features_per_frame = self.hist_size + self.lbp_size + self.edge_size
        self.num_frames = 2
        self.features = np.random.rand(
            self.num_frames, self.total_features_per_frame
        ).astype(np.float32)

        self.mock_cv2 = MagicMock()
        self.mock_cv2.VideoWriter_fourcc.return_value = 'mp4v'
        self.mock_writer = MagicMock()
        self.mock_cv2.VideoWriter.return_value = self.mock_writer
        self.mock_cv2.normalize.side_effect = lambda src, dst, alpha, beta, norm_type, dtype: (src * 255).astype(np.uint8)
        self.mock_cv2.addWeighted.side_effect = lambda src1, alpha, src2, beta, gamma: src1
        self.mock_cv2.cvtColor.side_effect = lambda src, code: src

    @patch('definers.tmp', return_value="/fake/video.mp4")
    def test_successful_video_generation(self, mock_tmp):
        with patch.dict('sys.modules', {'cv2': self.mock_cv2}):
            result = features_to_video(self.features, video_shape=self.video_shape)

        self.assertEqual(result, "/fake/video.mp4")
        self.mock_cv2.VideoWriter.assert_called_with(
            "/fake/video.mp4", 'mp4v', 24, (self.width, self.height)
        )
        self.assertEqual(self.mock_writer.write.call_count, self.num_frames)
        self.mock_writer.release.assert_called_once()

    @patch('definers.tmp', return_value="/fake/video.mp4")
    def test_exception_during_processing(self, mock_tmp):
        self.mock_cv2.normalize.side_effect = Exception("Test processing exception")
        with patch.dict('sys.modules', {'cv2': self.mock_cv2}):
            result = features_to_video(self.features, video_shape=self.video_shape)
        self.assertFalse(result)

    def test_empty_features_input(self):
        empty_features = np.array([])
        result = features_to_video(empty_features)
        self.assertFalse(result)

    def test_none_features_input(self):
        result = features_to_video(None)
        self.assertFalse(result)

if __name__ == "__main__":
    unittest.main()
