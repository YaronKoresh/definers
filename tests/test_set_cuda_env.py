import unittest
from unittest.mock import patch, call
import os
from definers import set_cuda_env

class TestSetCudaEnv(unittest.TestCase):

    @patch('definers.log')
    @patch('definers.paths')
    @patch.dict(os.environ, {}, clear=True)
    def test_cuda_paths_found(self, mock_paths, mock_log):
        mock_paths.side_effect = [
            ["/usr/local/cuda-12.2/"],
            ["/usr/local/cuda-12.2/lib64/"]
        ]
        set_cuda_env()
        self.assertEqual(os.environ["CUDA_PATH"], "/usr/local/cuda-12.2/")
        self.assertEqual(os.environ["LD_LIBRARY_PATH"], "/usr/local/cuda-12.2/lib64/")
        expected_calls = [
            call("CUDA_PATH", "/usr/local/cuda-12.2/", status=True),
            call("LD_LIBRARY_PATH", "/usr/local/cuda-12.2/lib64/", status=True)
        ]
        mock_log.assert_has_calls(expected_calls)

    @patch('definers.log')
    @patch('definers.paths')
    @patch.dict(os.environ, {}, clear=True)
    def test_no_cuda_paths_found(self, mock_paths, mock_log):
        mock_paths.return_value = []
        set_cuda_env()
        self.assertNotIn("CUDA_PATH", os.environ)
        self.assertNotIn("LD_LIBRARY_PATH", os.environ)
        mock_log.assert_called_once_with("Cuda not found", "Failed setting CUDA environment", status=False)

    @patch('definers.log')
    @patch('definers.paths')
    @patch.dict(os.environ, {}, clear=True)
    def test_only_cuda_path_found(self, mock_paths, mock_log):
        mock_paths.side_effect = [
            ["/opt/cuda/"],
            []
        ]
        set_cuda_env()
        self.assertNotIn("CUDA_PATH", os.environ)
        self.assertNotIn("LD_LIBRARY_PATH", os.environ)
        mock_log.assert_called_once_with("Cuda not found", "Failed setting CUDA environment", status=False)

    @patch('definers.log')
    @patch('definers.paths')
    @patch.dict(os.environ, {}, clear=True)
    def test_only_ld_library_path_found(self, mock_paths, mock_log):
        mock_paths.side_effect = [
            [],
            ["/opt/cuda/lib/"]
        ]
        set_cuda_env()
        self.assertNotIn("CUDA_PATH", os.environ)
        self.assertNotIn("LD_LIBRARY_PATH", os.environ)
        mock_log.assert_called_once_with("Cuda not found", "Failed setting CUDA environment", status=False)

if __name__ == '__main__':
    unittest.main()
