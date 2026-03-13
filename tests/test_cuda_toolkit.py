import unittest
from unittest.mock import call, patch

from definers import cuda_toolkit


class TestCudaToolkit(unittest.TestCase):
    @patch("definers.run")
    @patch("definers.permit")
    @patch("definers.directory")
    def test_cuda_toolkit_execution_flow(
        self, mock_directory, mock_permit, mock_run
    ):
        cuda_toolkit()
        expected_directory_calls = [
            call("/usr/share/keyrings/"),
            call("/etc/modprobe.d/"),
        ]
        mock_directory.assert_has_calls(
            expected_directory_calls, any_order=True
        )
        expected_permit_calls = [
            call("/tmp"),
            call("/usr/bin"),
            call("/usr/lib"),
            call("/usr/local"),
            call("/usr/share/keyrings/cuda-archive-keyring.gpg"),
            call("/etc/apt/sources.list.d/CUDA.list"),
        ]
        mock_permit.assert_has_calls(expected_permit_calls, any_order=True)
        expected_run_calls = [
            call(["apt-get", "update"]),
            call(["apt-get", "install", "-y", "curl"]),
            call(
                [
                    "bash",
                    "-lc",
                    '\n        export PATH=/sbin:$PATH\n        apt-get update\n        apt-get purge nvidia-*\n        echo "blacklist nouveau" > /etc/modprobe.d/blacklist-nouveau.conf\n        echo "options nouveau modeset=0" >> /etc/modprobe.d/blacklist-nouveau.conf\n        apt-get install -y --reinstall dkms\n        apt-get install -f\n        curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb > /usr/share/keyrings/cuda.deb\n        cd /usr/share/keyrings/\n        ar vx cuda.deb\n        tar xvf data.tar.xz\n        mv /usr/share/keyrings/usr/share/keyrings/cuda-archive-keyring.gpg /usr/share/keyrings/cuda-archive-keyring.gpg\n        rm -r /usr/share/keyrings/usr/\n        rm -r /usr/share/keyrings/etc/\n        echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /" > /etc/apt/sources.list.d/CUDA.list\n    ',
                ]
            ),
            call(
                [
                    "bash",
                    "-lc",
                    "\n        apt-get update\n        apt-get install -y cuda-toolkit\n    ",
                ]
            ),
        ]
        mock_run.assert_has_calls(expected_run_calls)


if __name__ == "__main__":
    unittest.main()
