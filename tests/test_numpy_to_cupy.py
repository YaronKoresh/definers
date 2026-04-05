import unittest

import numpy as np

from definers.application_data.arrays import numpy_to_cupy


class TestNumpyToCupy(unittest.TestCase):
    def test_numpy_array_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = numpy_to_cupy(arr)
        result_np = np.array(result)
        np.testing.assert_array_equal(result_np, arr)

    def test_2d_array(self):
        arr = np.zeros((2, 3))
        result = numpy_to_cupy(arr)
        result_np = np.array(result)
        self.assertEqual(result_np.shape, (2, 3))

    def test_integer_array(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = numpy_to_cupy(arr)
        result_np = np.array(result)
        np.testing.assert_array_equal(result_np, arr)

    def test_result_values_preserved(self):
        arr = np.array([10.5, 20.5, 30.5])
        result = numpy_to_cupy(arr)
        result_np = np.array(result)
        np.testing.assert_allclose(result_np, arr)


if __name__ == "__main__":
    unittest.main()
