import unittest
from unittest.mock import patch, call
from definers import install_faiss

class TestInstallFaiss(unittest.TestCase):

    @patch('definers.importable', return_value=True)
    def test_faiss_already_installed(self, mock_importable):
        result = install_faiss()
        self.assertFalse(result)
        mock_importable.assert_called_once_with("faiss")

    @patch('definers.importable', return_value=False)
    @patch('definers.subprocess.run')
    @patch('definers.cwd')
    @patch('definers.sys')
    def test_successful_installation(self, mock_sys, mock_cwd, mock_subprocess_run, mock_importable):
        mock_sys.executable = "/path/to/python"
        mock_sys.prefix = "/path/to/prefix"
        mock_sys.version_info.major = 3
        mock_sys.version_info.minor = 10
        mock_cwd.return_value.__enter__.return_value = "_faiss_"
        
        install_faiss()

        self.assertEqual(mock_subprocess_run.call_count, 5)
        
        calls = [
            call(["git", "clone", "https://github.com/facebookresearch/faiss.git", "_faiss_"], check=True),
            call([
                "cmake", "-B", "_faiss_/build", "-DBUILD_TESTING=OFF", "-DCMAKE_BUILD_TYPE=Release",
                "-DFAISS_ENABLE_C_API=ON", "-DFAISS_ENABLE_GPU=ON", "-DFAISS_ENABLE_PYTHON=ON",
                f"-DPython_EXECUTABLE=/path/to/python",
                f"-DPython_INCLUDE_DIR=/path/to/prefix/include/python3.10",
                f"-DPython_LIBRARY=/path/to/prefix/lib/libpython3.10.so",
                f"-DPython_NumPy_INCLUDE_DIRS=/path/to/prefix/lib/python3.10/site-packages/numpy/core/include",
                "."
            ], check=True),
            call(["make", "-C", "_faiss_/build", "-j16", "faiss"], check=True),
            call(["make", "-C", "_faiss_/build", "-j16", "swigfaiss"], check=True),
            call(["/path/to/python", "-m", "pip", "install", "."], cwd="_faiss_/build/faiss/python", check=True)
        ]
        mock_subprocess_run.assert_has_calls(calls)

    @patch('definers.importable', return_value=False)
    @patch('definers.subprocess.run', side_effect=Exception("Test Exception"))
    @patch('builtins.print')
    def test_installation_failure(self, mock_print, mock_subprocess_run, mock_importable):
        install_faiss()
        mock_print.assert_any_call("An unexpected error occurred: Test Exception")

    @patch('definers.importable', return_value=False)
    @patch('definers.subprocess.run', side_effect=FileNotFoundError("git not found"))
    @patch('builtins.print')
    def test_git_not_found(self, mock_print, mock_subprocess_run, mock_importable):
        install_faiss()
        mock_print.assert_any_call("File not found error: git not found")

if __name__ == '__main__':
    unittest.main()
