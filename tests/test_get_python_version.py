import sys
import unittest
from unittest.mock import MagicMock, patch

from definers import get_python_version


class TestGetPythonVersion(unittest.TestCase):
    def test_get_python_version_successfully(self):
        expected_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.assertEqual(get_python_version(), expected_version)

    @patch("definers.sys")
    def test_get_python_version_missing_micro(self, mock_sys):
        mock_version_info = MagicMock(spec=["major", "minor"])
        mock_version_info.major = 3
        mock_version_info.minor = 11
        del mock_version_info.micro

        mock_sys.version_info = mock_version_info

        self.assertEqual(get_python_version(), "3.11.0")

    @patch("definers.sys")
    def test_get_python_version_missing_major(self, mock_sys):
        mock_version_info = MagicMock(spec=["minor", "micro"])
        del mock_version_info.major

        mock_sys.version_info = mock_version_info

        self.assertIsNone(get_python_version())


if __name__ == "__main__":
    unittest.main()
