import unittest

import numpy as np

import definers._data as _data

_data.np = np
_data._np = np

from definers import guess_numpy_type


class TestGuessNumpyType(unittest.TestCase):
    def test_4d_array_is_video(self):
        arr = np.zeros((10, 64, 64, 3), dtype=np.float32)
        self.assertEqual(guess_numpy_type(arr), "video")

    def test_3d_array_is_image(self):
        arr = np.zeros((64, 64, 3), dtype=np.float32)
        self.assertEqual(guess_numpy_type(arr), "image")

    def test_1d_float_array_is_audio(self):
        arr = np.array([0.1, 0.2, -0.3, 0.4], dtype=np.float32)
        self.assertEqual(guess_numpy_type(arr), "audio")

    def test_2d_float_is_audio(self):
        arr = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        result = guess_numpy_type(arr)
        self.assertEqual(result, "audio")


if __name__ == "__main__":
    unittest.main()
