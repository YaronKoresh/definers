import site
import sys
import unittest
from unittest.mock import patch

from definers import add_path


class TestAddPath(unittest.TestCase):

    @patch("definers.permit")
    @patch("site.addsitedir")
    def test_add_new_path(self, mock_addsitedir, mock_permit):
        test_path = "/new/test/path"
        original_sys_path = sys.path[:]

        try:
            sys.path = [
                p for p in original_sys_path if p != test_path
            ]

            add_path(test_path)

            mock_permit.assert_called_once_with(test_path)
            self.assertIn(test_path, sys.path)
            mock_addsitedir.assert_called_once_with(test_path)
        finally:
            sys.path = original_sys_path

    @patch("definers.permit")
    @patch("site.addsitedir")
    def test_add_existing_path(self, mock_addsitedir, mock_permit):
        test_path = "/existing/test/path"
        original_sys_path = sys.path[:]

        try:
            if test_path not in sys.path:
                sys.path.append(test_path)

            initial_path_length = len(sys.path)

            add_path(test_path)

            mock_permit.assert_not_called()
            mock_addsitedir.assert_not_called()
            self.assertEqual(len(sys.path), initial_path_length)
        finally:
            sys.path = original_sys_path

    @patch("definers.permit")
    @patch("site.addsitedir")
    def test_add_empty_path(self, mock_addsitedir, mock_permit):
        test_path = ""
        original_sys_path = sys.path[:]

        try:
            sys.path = [
                p for p in original_sys_path if p != test_path
            ]

            add_path(test_path)

            mock_permit.assert_called_once_with(test_path)
            self.assertIn(test_path, sys.path)
            mock_addsitedir.assert_called_once_with(test_path)
        finally:
            sys.path = original_sys_path


if __name__ == "__main__":
    unittest.main()
