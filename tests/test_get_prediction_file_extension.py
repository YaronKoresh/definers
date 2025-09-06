import unittest

from definers import get_prediction_file_extension


class TestGetPredictionFileExtension(unittest.TestCase):

    def test_video_extension(self):
        self.assertEqual(
            get_prediction_file_extension("video"), "mp4"
        )

    def test_image_extension(self):
        self.assertEqual(
            get_prediction_file_extension("image"), "png"
        )

    def test_audio_extension(self):
        self.assertEqual(
            get_prediction_file_extension("audio"), "wav"
        )

    def test_text_extension(self):
        self.assertEqual(get_prediction_file_extension("text"), "txt")

    def test_unknown_extension(self):
        self.assertEqual(
            get_prediction_file_extension("unknown_type"), "data"
        )
        self.assertEqual(get_prediction_file_extension(""), "data")
        self.assertEqual(get_prediction_file_extension(None), "data")

    def test_case_insensitivity_though_not_required(self):
        self.assertEqual(
            get_prediction_file_extension("Video"), "mp4"
        )
        self.assertEqual(
            get_prediction_file_extension("IMAGE"), "png"
        )


if __name__ == "__main__":
    unittest.main()
