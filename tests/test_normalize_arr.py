import unittest

import numpy as np

from definers import normalize_arr


class TestNormalizeArr(unittest.TestCase):
    def test_simple_array(self):
        arr = np.array([0.0, 5.0, 10.0])
        result = normalize_arr(arr)
        self.assertAlmostEqual(result.min(), 0.0)
        self.assertAlmostEqual(result.max(), 1.0)

    def test_already_normalized(self):
        arr = np.array([0.0, 0.5, 1.0])
        result = normalize_arr(arr)
        np.testing.assert_array_almost_equal(result, arr)

    def test_constant_array(self):
        arr = np.array([5.0, 5.0, 5.0])
        result = normalize_arr(arr)
        np.testing.assert_array_equal(result, np.zeros_like(arr))


if __name__ == "__main__":
    unittest.main()
