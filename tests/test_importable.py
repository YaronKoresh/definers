import unittest
from unittest.mock import patch

from definers import importable


class TestImportable(unittest.TestCase):

    @patch("definers.run")
    def test_importable_returns_true_for_successful_import(
        self, mock_run
    ):
        mock_run.return_value = []
        self.assertTrue(importable("os"))
        mock_run.assert_called_once_with(
            'python -c "import os"', silent=True
        )

    @patch("definers.run")
    def test_importable_returns_false_for_failed_import(
        self, mock_run
    ):
        mock_run.return_value = False
        self.assertFalse(importable("non_existent_package"))
        mock_run.assert_called_once_with(
            'python -c "import non_existent_package"', silent=True
        )

    @patch("definers.run")
    def test_importable_handles_run_returning_output(self, mock_run):
        mock_run.return_value = ["some output"]
        self.assertTrue(importable("sys"))

    @patch("definers.run")
    def test_importable_with_empty_string(self, mock_run):
        mock_run.return_value = False
        self.assertFalse(importable(""))

    @patch("definers.run")
    def test_importable_with_complex_name(self, mock_run):
        mock_run.return_value = True
        self.assertTrue(importable("unittest.mock"))
        mock_run.assert_called_once_with(
            'python -c "import unittest.mock"', silent=True
        )


if __name__ == "__main__":
    unittest.main()
