import unittest

import numpy as np

from definers.data import numpy_to_str


class TestNumpyToStr(unittest.TestCase):
    def test_simple_string_array(self):
        arr = np.array(["hello", "world"])
        result = numpy_to_str(arr)
        self.assertEqual(result, "hello world")

    def test_single_element(self):
        arr = np.array(["single"])
        result = numpy_to_str(arr)
        self.assertEqual(result, "single")

    def test_2d_array(self):
        arr = np.array([["a", "b"], ["c", "d"]])
        result = numpy_to_str(arr)
        self.assertEqual(result, "a b c d")

    def test_returns_string(self):
        arr = np.array(["test"])
        result = numpy_to_str(arr)
        self.assertIsInstance(result, str)

    def test_numeric_array_to_str(self):
        arr = np.array([1, 2, 3])
        result = numpy_to_str(arr)
        self.assertEqual(result, "1 2 3")


if __name__ == "__main__":
    unittest.main()
