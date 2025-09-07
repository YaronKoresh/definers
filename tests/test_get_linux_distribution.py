import subprocess
import unittest
from unittest.mock import MagicMock, patch

from definers import get_linux_distribution


class TestGetLinuxDistribution(unittest.TestCase):
    @patch("definers.subprocess.run")
    def test_with_lsb_release(self, mock_run):
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=["apt-get", "update"], returncode=0
            ),
            subprocess.CompletedProcess(
                args=["apt-get", "install", "-y", "lsb_release"],
                returncode=0,
            ),
            subprocess.CompletedProcess(
                args=["lsb_release", "-a"],
                returncode=0,
                stdout="Distributor ID:\tUbuntu\nRelease:\t20.04\n",
                stderr="",
            ),
        ]
        distro, release = get_linux_distribution()
        self.assertEqual(distro, "ubuntu")
        self.assertEqual(release, "20.04")

    @patch("definers.subprocess.run", side_effect=FileNotFoundError)
    @patch("builtins.open")
    def test_with_os_release(self, mock_open, mock_run):
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = (
            'NAME="Ubuntu"\nVERSION_ID="20.04"'
        )
        mock_open.return_value = mock_file

        distro, release = get_linux_distribution()
        self.assertEqual(distro, "Ubuntu")
        self.assertEqual(release, "20.04")

    @patch("definers.subprocess.run", side_effect=FileNotFoundError)
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_no_method_works(self, mock_open, mock_run):
        distro, release = get_linux_distribution()
        self.assertIsNone(distro)
        self.assertIsNone(release)


if __name__ == "__main__":
    unittest.main()
