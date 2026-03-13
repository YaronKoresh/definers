import unittest

import numpy as np

import definers.data as data

data.np = np
from definers import numpy_to_list


class TestNumpyToList(unittest.TestCase):
    def test_single_1d_array(self):
        arr = np.array([1, 2, 3])
        result = numpy_to_list([arr])
        self.assertEqual(result, [1, 2, 3])

    def test_multiple_1d_arrays(self):
        a = np.array([1, 2])
        b = np.array([3, 4])
        result = numpy_to_list([a, b])
        self.assertEqual(result, [1, 2, 3, 4])

    def test_returns_python_list(self):
        arr = np.array([5, 6, 7])
        result = numpy_to_list([arr])
        self.assertIsInstance(result, list)

    def test_2d_array_flattened(self):
        arr = np.array([[1, 2], [3, 4]])
        result = numpy_to_list([arr])
        self.assertEqual(sorted(result), [1, 2, 3, 4])

    def test_float_array(self):
        arr = np.array([1.5, 2.5], dtype=np.float32)
        result = numpy_to_list([arr])
        self.assertAlmostEqual(result[0], 1.5, places=5)
        self.assertAlmostEqual(result[1], 2.5, places=5)


if __name__ == "__main__":
    unittest.main()
