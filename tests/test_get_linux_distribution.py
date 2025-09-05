import unittest
from unittest.mock import patch, mock_open, MagicMock
import subprocess
from definers import get_linux_distribution

class TestGetLinuxDistribution(unittest.TestCase):

    @patch('subprocess.run')
    def test_get_linux_distribution_with_lsb_release(self, mock_run):
        mock_process = MagicMock()
        mock_process.stdout = "Distributor ID:	Ubuntu\nRelease:	22.04\n"
        mock_run.return_value = mock_process
        
        distro, release = get_linux_distribution()
        self.assertEqual(distro, "ubuntu")
        self.assertEqual(release, "22.04")
        mock_run.assert_any_call(['lsb_release', '-a'], capture_output=True, text=True, check=True)

    @patch('subprocess.run', side_effect=FileNotFoundError)
    @patch('builtins.open', new_callable=mock_open, read_data='NAME="Ubuntu"\nVERSION_ID="20.04"')
    def test_get_linux_distribution_with_os_release(self, mock_file, mock_run):
        distro, release = get_linux_distribution()
        self.assertEqual(distro, "Ubuntu")
        self.assertEqual(release, "20.04")

    @patch('subprocess.run', side_effect=FileNotFoundError)
    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_get_linux_distribution_no_methods_available(self, mock_open, mock_run):
        distro, release = get_linux_distribution()
        self.assertIsNone(distro)
        self.assertIsNone(release)
        
    @patch('subprocess.run')
    def test_get_linux_distribution_lsb_release_fails_gracefully(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")
        with patch('builtins.open', new_callable=mock_open, read_data='NAME="CentOS Linux"\nVERSION_ID="8"') as mock_file:
            distro, release = get_linux_distribution()
            self.assertEqual(distro, "CentOS Linux")
            self.assertEqual(release, "8")

    @patch('subprocess.run', side_effect=FileNotFoundError)
    @patch('builtins.open', new_callable=mock_open, read_data='UNEXPECTED_FORMAT')
    def test_get_linux_distribution_os_release_bad_format(self, mock_file, mock_run):
        distro, release = get_linux_distribution()
        self.assertIsNone(distro)
        self.assertIsNone(release)

    @patch('subprocess.run')
    def test_get_linux_distribution_lsb_release_bad_format(self, mock_run):
        mock_process = MagicMock()
        mock_process.stdout = "Some other output"
        mock_run.return_value = mock_process
        
        with patch('builtins.open', side_effect=FileNotFoundError):
            distro, release = get_linux_distribution()
            self.assertIsNone(distro)
            self.assertIsNone(release)

if __name__ == '__main__':
    unittest.main()
