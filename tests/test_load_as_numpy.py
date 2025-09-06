import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import imageio.v2 as iio
import numpy as np
import pandas as pd
import soundfile as sf

from definers import load_as_numpy


class TestLoadAsNumpy(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.mock_features = np.array([0.1, 0.2, 0.3])

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _create_dummy_file(self, file_name, content_writer, *args):
        path = os.path.join(self.test_dir, file_name)
        content_writer(path, *args)
        return path

    def _write_wav(self, path):
        sf.write(path, np.random.randn(22050), 22050)

    def _write_csv(self, path):
        pd.DataFrame({"a": [1], "b": [2]}).to_csv(path, index=False)

    def _write_txt(self, path):
        with open(path, "w") as f:
            f.write("test")

    def _write_png(self, path):
        iio.imwrite(path, np.zeros((10, 10, 3), dtype=np.uint8))

    def _write_mp4(self, path):
        writer = iio.get_writer(path, fps=1)
        writer.append_data(np.zeros((10, 10, 3), dtype=np.uint8))
        writer.close()

    @patch(
        "definers.extract_audio_features",
        return_value=np.array([1, 2, 3]),
    )
    @patch("definers.sox.Transformer")
    def test_load_audio_wav_no_training(self, mock_sox, mock_extract):
        mock_transformer = MagicMock()
        mock_sox.return_value = mock_transformer
        wav_path = self._create_dummy_file(
            "test.wav", self._write_wav
        )
        result = load_as_numpy(wav_path, training=False)
        self.assertIsInstance(result, np.ndarray)
        mock_transformer.build_file.assert_called_once()
        mock_extract.assert_called_once()

    @patch(
        "definers.extract_audio_features",
        return_value=np.array([1, 2, 3]),
    )
    @patch("definers.split_mp3", return_value=("some_dir", 1))
    @patch("definers.remove_silence")
    @patch("definers.sox.Transformer")
    def test_load_audio_mp3_training(
        self, mock_sox, mock_silence, mock_split, mock_extract
    ):
        mock_transformer = MagicMock()
        mock_sox.return_value = mock_transformer
        mp3_path = self._create_dummy_file(
            "test.mp3", self._write_wav
        )
        result = load_as_numpy(mp3_path, training=True)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], np.ndarray)
        mock_transformer.build_file.assert_called_once()
        mock_silence.assert_called_once()
        mock_split.assert_called_once()

    def test_load_csv(self):
        csv_path = self._create_dummy_file(
            "test.csv", self._write_csv
        )
        result = load_as_numpy(csv_path)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, 2))
        np.testing.assert_array_equal(result, np.array([[1, 2]]))

    @patch("pandas.read_excel")
    def test_load_xlsx(self, mock_read_excel):
        mock_read_excel.return_value = pd.DataFrame({"a": [1]})
        xlsx_path = os.path.join(self.test_dir, "test.xlsx")
        with open(xlsx_path, "w") as f:
            f.write("")
        result = load_as_numpy(xlsx_path)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, 1))

    @patch("definers.extract_text_features")
    def test_load_txt(self, mock_extract_text):
        mock_extract_text.return_value = self.mock_features
        txt_path = self._create_dummy_file(
            "test.txt", self._write_txt
        )
        result = load_as_numpy(txt_path)
        mock_extract_text.assert_called_with("test")
        np.testing.assert_array_equal(result, self.mock_features)

    @patch("definers.extract_image_features")
    @patch("definers.resize_image")
    def test_load_image(self, mock_resize, mock_extract):
        mock_resize.return_value = np.zeros((1024, 1024, 3))
        mock_extract.return_value = self.mock_features
        img_path = self._create_dummy_file(
            "test.png", self._write_png
        )
        result = load_as_numpy(img_path)
        mock_resize.assert_called_once()
        mock_extract.assert_called_once()
        np.testing.assert_array_equal(result, self.mock_features)

    @patch("definers.extract_video_features")
    @patch("definers.convert_video_fps")
    @patch("definers.resize_video")
    def test_load_video(self, mock_resize, mock_fps, mock_extract):
        mock_resize.return_value = "resized.mp4"
        mock_fps.return_value = "resized_fps.mp4"
        mock_extract.return_value = self.mock_features
        vid_path = self._create_dummy_file(
            "test.mov", self._write_mp4
        )
        result = load_as_numpy(vid_path)
        mock_resize.assert_called_with(vid_path, 1024, 1024)
        mock_fps.assert_called_with("resized.mp4", 24)
        mock_extract.assert_called_with("resized_fps.mp4")
        np.testing.assert_array_equal(result, self.mock_features)

    @patch("definers.catch")
    def test_invalid_path_format(self, mock_catch):
        result = load_as_numpy("invalidpath")
        self.assertIsNone(result)

    @patch("definers.catch")
    def test_non_existent_file(self, mock_catch):
        result = load_as_numpy("nonexistent.txt")
        self.assertIsNone(result)

    @patch("definers.catch")
    @patch("pandas.read_csv", side_effect=Exception("Read Error"))
    def test_corrupt_csv(self, mock_read_csv, mock_catch):
        csv_path = self._create_dummy_file(
            "corrupt.csv", lambda p: open(p, "w").close()
        )
        result = load_as_numpy(csv_path)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
