import unittest

import numpy as np

from definers.data import get_max_shapes


class TestGetMaxShapes(unittest.TestCase):
    def test_single_array(self):
        a = np.zeros((3, 4))
        result = get_max_shapes(a)
        self.assertEqual(result, [3, 4])

    def test_two_arrays_same_shape(self):
        a = np.zeros((2, 5))
        b = np.zeros((2, 5))
        result = get_max_shapes(a, b)
        self.assertEqual(result, [2, 5])

    def test_two_arrays_different_shapes(self):
        a = np.zeros((3, 4))
        b = np.zeros((5, 2))
        result = get_max_shapes(a, b)
        self.assertEqual(result, [5, 4])

    def test_1d_arrays(self):
        a = np.zeros((3,))
        b = np.zeros((7,))
        result = get_max_shapes(a, b)
        self.assertEqual(result, [7])

    def test_3d_arrays(self):
        a = np.zeros((2, 3, 4))
        b = np.zeros((5, 1, 6))
        result = get_max_shapes(a, b)
        self.assertEqual(result, [5, 3, 6])

    def test_mismatched_ndim(self):
        a = np.zeros((3, 4))
        b = np.zeros((2,))
        result = get_max_shapes(a, b)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], 3)


if __name__ == "__main__":
    unittest.main()
