import unittest

from definers.path_utils import unique


class TestUnique(unittest.TestCase):
    def test_removes_duplicates(self):
        self.assertEqual(unique([1, 2, 2, 3, 3, 3]), [1, 2, 3])

    def test_empty_list(self):
        self.assertEqual(unique([]), [])

    def test_no_duplicates(self):
        self.assertEqual(unique([1, 2, 3]), [1, 2, 3])

    def test_all_same(self):
        self.assertEqual(unique([5, 5, 5]), [5])

    def test_strings(self):
        self.assertEqual(unique(["a", "b", "a", "c"]), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
