import unittest
from unittest.mock import patch

from definers import importable


class TestImportable(unittest.TestCase):
    @patch("importlib.util.find_spec")
    def test_importable_returns_true_for_successful_import(
        self, mock_find_spec
    ):
        mock_find_spec.return_value = object()
        self.assertTrue(importable("os"))
        mock_find_spec.assert_called_once_with("os")

    @patch("importlib.util.find_spec")
    def test_importable_returns_false_for_failed_import(self, mock_find_spec):
        mock_find_spec.return_value = None
        self.assertFalse(importable("non_existent_package"))
        mock_find_spec.assert_called_once_with("non_existent_package")

    @patch("importlib.util.find_spec")
    def test_importable_handles_complex_name(self, mock_find_spec):
        mock_find_spec.return_value = object()
        self.assertTrue(importable("sys"))

    @patch("importlib.util.find_spec")
    def test_importable_with_empty_string(self, mock_find_spec):
        self.assertFalse(importable(""))
        mock_find_spec.assert_not_called()

    @patch("importlib.util.find_spec")
    def test_importable_with_dotted_name(self, mock_find_spec):
        mock_find_spec.return_value = object()
        self.assertTrue(importable("unittest.mock"))
        mock_find_spec.assert_called_once_with("unittest.mock")

    @patch("importlib.util.find_spec", side_effect=Exception)
    def test_importable_handles_find_spec_exception(self, mock_find_spec):
        self.assertFalse(importable("bad.module"))
        mock_find_spec.assert_called_once_with("bad.module")


if __name__ == "__main__":
    unittest.main()
