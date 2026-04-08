import unittest
from unittest.mock import patch

from definers.data.loaders import select_rows
from tests.optional_dependency_stubs import (
    FakeDataset,
    build_fake_datasets_module,
)


class TestSelectRows(unittest.TestCase):
    def setUp(self):
        self.data = {"col1": [1, 2, 3, 4, 5], "col2": ["A", "B", "C", "D", "E"]}
        self.dataset = FakeDataset.from_dict(self.data)

    def test_selects_a_slice_of_rows(self):
        with patch.dict(
            "sys.modules", {"datasets": build_fake_datasets_module()}
        ):
            result = select_rows(self.dataset, 1, 4)
        self.assertEqual(len(result), 3)
        self.assertEqual(result["col1"], [2, 3, 4])
        self.assertEqual(result["col2"], ["B", "C", "D"])

    def test_selects_from_start(self):
        with patch.dict(
            "sys.modules", {"datasets": build_fake_datasets_module()}
        ):
            result = select_rows(self.dataset, 0, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result["col1"], [1, 2])
        self.assertEqual(result["col2"], ["A", "B"])

    def test_selects_until_end(self):
        with patch.dict(
            "sys.modules", {"datasets": build_fake_datasets_module()}
        ):
            result = select_rows(self.dataset, 3, 5)
        self.assertEqual(len(result), 2)
        self.assertEqual(result["col1"], [4, 5])
        self.assertEqual(result["col2"], ["D", "E"])

    def test_selects_single_row(self):
        with patch.dict(
            "sys.modules", {"datasets": build_fake_datasets_module()}
        ):
            result = select_rows(self.dataset, 2, 3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["col1"], [3])
        self.assertEqual(result["col2"], ["C"])

    def test_handles_empty_slice(self):
        with patch.dict(
            "sys.modules", {"datasets": build_fake_datasets_module()}
        ):
            result = select_rows(self.dataset, 2, 2)
        self.assertEqual(len(result), 0)
        self.assertEqual(result["col1"], [])
        self.assertEqual(result["col2"], [])


if __name__ == "__main__":
    unittest.main()
