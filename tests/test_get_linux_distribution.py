import unittest
from unittest.mock import patch, mock_open, MagicMock
import subprocess
from definers import get_linux_distribution

class TestGetLinuxDistribution(unittest.TestCase):

    @patch('subprocess.run')
    def test_with_lsb_release(self, mock_run):
        mock_run.side_effect = [
            MagicMock(check=True), 
            MagicMock(stdout="Distributor ID:	Ubuntu\nRelease:	20.04\n", check=True)
        ]
        distro, release = get_linux_distribution()
        self.assertEqual(distro, 'ubuntu')
        self.assertEqual(release, '20.04')
        self.assertEqual(mock_run.call_count, 2)

    @patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'cmd'))
    @patch('builtins.open', new_callable=mock_open, read_data='NAME="Ubuntu"\nVERSION_ID="20.04"')
    def test_with_os_release_fallback(self, mock_file, mock_run):
        distro, release = get_linux_distribution()
        self.assertEqual(distro, 'Ubuntu')
        self.assertEqual(release, '20.04')
        mock_file.assert_called_with("/etc/os-release", "r")

    @patch('subprocess.run', side_effect=FileNotFoundError)
    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_no_methods_succeed(self, mock_file, mock_run):
        distro, release = get_linux_distribution()
        self.assertIsNone(distro)
        self.assertIsNone(release)

    @patch('subprocess.run')
    def test_malformed_lsb_release_output(self, mock_run):
        mock_run.side_effect = [
            MagicMock(check=True),
            MagicMock(stdout="Some other output", check=True)
        ]
        with patch('builtins.open', side_effect=FileNotFoundError):
            distro, release = get_linux_distribution()
            self.assertIsNone(distro)
            self.assertIsNone(release)

    @patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'cmd'))
    @patch('builtins.open', new_callable=mock_open, read_data='INVALID_CONTENT')
    def test_malformed_os_release_content(self, mock_file, mock_run):
        distro, release = get_linux_distribution()
        self.assertIsNone(distro)
        self.assertIsNone(release)

if __name__ == '__main__':
    unittest.main()
