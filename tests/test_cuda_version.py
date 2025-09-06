import subprocess
import unittest
from unittest.mock import Mock, patch

from definers import cuda_version


class TestCudaVersion(unittest.TestCase):

    @patch("subprocess.run")
    def test_cuda_version_present(self, mock_subprocess_run):
        mock_process = Mock()
        mock_process.stdout = "nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) 2005-2023 NVIDIA Corporation\nBuilt on Mon_Apr_  3_17:16:06_PDT_2023\nCuda compilation tools, release 12.2, V12.2.140\nBuild cuda_12.2.r12.2/compiler.32688072_0"
        mock_process.check_returncode.return_value = None
        mock_subprocess_run.return_value = mock_process

        version = cuda_version()
        self.assertEqual(version, "12.2")
        mock_subprocess_run.assert_called_once_with(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch("subprocess.run")
    def test_cuda_version_different_format(self, mock_subprocess_run):
        mock_process = Mock()
        mock_process.stdout = "Cuda compilation tools, release 11.8, V11.8.89\nBuild cuda_11.8.r11.8/compiler.31833905_0"
        mock_process.check_returncode.return_value = None
        mock_subprocess_run.return_value = mock_process

        version = cuda_version()
        self.assertEqual(version, "11.8")

    @patch("subprocess.run")
    def test_no_cuda_version_in_output(self, mock_subprocess_run):
        mock_process = Mock()
        mock_process.stdout = (
            "Some other command output without version info"
        )
        mock_process.check_returncode.return_value = None
        mock_subprocess_run.return_value = mock_process

        version = cuda_version()
        self.assertFalse(version)

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_nvcc_not_found(self, mock_subprocess_run):
        version = cuda_version()
        self.assertFalse(version)

    @patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "nvcc"),
    )
    def test_nvcc_command_fails(self, mock_subprocess_run):
        version = cuda_version()
        self.assertFalse(version)


if __name__ == "__main__":
    unittest.main()
