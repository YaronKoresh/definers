import unittest

import numpy as _numpy

import definers.data as data

data._np = _numpy
data.np = _numpy
from definers import pad_nested


class TestPadNested(unittest.TestCase):
    def test_flat_list_no_padding_needed(self):
        result = pad_nested([1, 2, 3], [3])
        self.assertEqual(result, [1, 2, 3])

    def test_flat_list_pads_to_length(self):
        result = pad_nested([1, 2], [5])
        self.assertEqual(result, [1, 2, 0, 0, 0])

    def test_flat_list_custom_fill_value(self):
        result = pad_nested([1], [4], fill_value=-1)
        self.assertEqual(result, [1, -1, -1, -1])

    def test_empty_list_returns_fill_values(self):
        result = pad_nested([], [3])
        self.assertEqual(result, [0, 0, 0])

    def test_nested_list_pads_inner(self):
        result = pad_nested([[1, 2], [3]], [2, 3])
        self.assertEqual(result, [[1, 2, 0], [3, 0, 0]])

    def test_nested_list_pads_outer(self):
        result = pad_nested([[1, 2]], [3, 2])
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], [1, 2])

    def test_nested_with_fill_value(self):
        result = pad_nested([[1]], [2, 3], fill_value=9)
        self.assertEqual(result[0], [1, 9, 9])
        self.assertEqual(result[1], [9, 9, 9])

    def test_numpy_array_input(self):
        arr = _numpy.array([1, 2, 3])
        result = pad_nested(arr, [5])
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0], 1)


if __name__ == "__main__":
    unittest.main()
