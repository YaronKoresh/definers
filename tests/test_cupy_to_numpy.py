import unittest

import numpy as np

from definers import cupy_to_numpy


class TestCupyToNumpy(unittest.TestCase):
    def test_numpy_array_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = cupy_to_numpy(arr)
        np.testing.assert_array_equal(result, arr)

    def test_list_passthrough(self):
        lst = [1, 2, 3]
        result = cupy_to_numpy(lst)
        self.assertEqual(result, lst)

    def test_scalar_passthrough(self):
        result = cupy_to_numpy(42)
        self.assertEqual(result, 42)

    def test_none_passthrough(self):
        result = cupy_to_numpy(None)
        self.assertIsNone(result)

    def test_string_passthrough(self):
        result = cupy_to_numpy("hello")
        self.assertEqual(result, "hello")

    def test_2d_array_passthrough(self):
        arr = np.zeros((3, 4))
        result = cupy_to_numpy(arr)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3, 4))


if __name__ == "__main__":
    unittest.main()
