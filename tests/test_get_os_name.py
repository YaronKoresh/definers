import unittest
from unittest.mock import patch
from definers import get_os_name

class TestGetOsName(unittest.TestCase):

    @patch('platform.system', return_value='Linux')
    def test_linux_os(self, mock_system):
        self.assertEqual(get_os_name(), 'linux')

    @patch('platform.system', return_value='Windows')
    def test_windows_os(self, mock_system):
        self.assertEqual(get_os_name(), 'windows')

    @patch('platform.system', return_value='Darwin')
    def test_darwin_os(self, mock_system):
        self.assertEqual(get_os_name(), 'darwin')

    @patch('platform.system', return_value='Java')
    def test_other_os(self, mock_system):
        self.assertEqual(get_os_name(), 'java')

    @patch('platform.system', return_value='')
    def test_empty_string_os(self, mock_system):
        self.assertEqual(get_os_name(), '')

if __name__ == '__main__':
    unittest.main()
