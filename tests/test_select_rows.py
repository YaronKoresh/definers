from definers import select_rows

from datasets import Dataset

import unittest
from unittest.mock import MagicMock

class TestSelectRows(unittest.TestCase):

    def setUp(self):
        self.data = {
            'col1': [1, 2, 3, 4, 5],
            'col2': ['A', 'B', 'C', 'D', 'E']
        }
        self.dataset = Dataset.from_dict(self.data)

    def test_selects_a_slice_of_rows(self):
        result = select_rows(self.dataset, 1, 4)
        self.assertEqual(len(result), 3)
        self.assertEqual(result['col1'], [2, 3, 4])
        self.assertEqual(result['col2'], ['B', 'C', 'D'])

    def test_selects_from_start(self):
        result = select_rows(self.dataset, 0, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result['col1'], [1, 2])
        self.assertEqual(result['col2'], ['A', 'B'])

    def test_selects_until_end(self):
        result = select_rows(self.dataset, 3, 5)
        self.assertEqual(len(result), 2)
        self.assertEqual(result['col1'], [4, 5])
        self.assertEqual(result['col2'], ['D', 'E'])

    def test_selects_single_row(self):
        result = select_rows(self.dataset, 2, 3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result['col1'], [3])
        self.assertEqual(result['col2'], ['C'])

    def test_handles_empty_slice(self):
        result = select_rows(self.dataset, 2, 2)
        self.assertEqual(len(result), 0)
        self.assertEqual(result['col1'], [])
        self.assertEqual(result['col2'], [])

if __name__ == '__main__':
    unittest.main()