import os
import unittest
from importlib import import_module
from unittest.mock import call, patch

import definers.os_utils as os_utils
import definers.path_utils as path_utils

if not hasattr(os_utils, "get_python_version"):
    os_utils.get_python_version = lambda: "3.10"
if not hasattr(os_utils, "get_linux_distribution"):
    os_utils.get_linux_distribution = lambda: "linux"

for _name, _value in {
    "normalize_path": lambda path: str(path),
    "full_path": lambda *parts: "/".join(
        str(part) for part in parts if str(part)
    ),
    "paths": lambda *patterns: [],
    "unique": lambda items: list(dict.fromkeys(items)),
    "cwd": lambda: ".",
    "parent_directory": lambda path: "",
    "path_end": lambda path: str(path).rsplit("/", 1)[-1].rsplit("\\", 1)[-1],
    "path_ext": lambda path: (
        "" if "." not in str(path) else "." + str(path).rsplit(".", 1)[-1]
    ),
    "path_name": lambda path: (
        str(path).rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0]
    ),
    "tmp": lambda *args, **kwargs: "/tmp/mock",
    "secure_path": lambda path, *args, **kwargs: path,
}.items():
    if not hasattr(path_utils, _name):
        setattr(path_utils, _name, _value)


def cuda_module():
    return import_module("definers.cuda")


class TestSetCudaEnv(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_cuda_paths_found(self):
        module = cuda_module()
        with (
            patch.object(module, "get_os_name", return_value="linux"),
            patch.object(module, "log") as mock_log,
            patch.object(module, "paths") as mock_paths,
        ):
            mock_paths.side_effect = [
                ["/usr/local/cuda-12.2/"],
                ["/usr/local/cuda-12.2/lib64/"],
            ]
            module.set_cuda_env()
        self.assertEqual(os.environ["CUDA_PATH"], "/usr/local/cuda-12.2/")
        self.assertEqual(
            os.environ["LD_LIBRARY_PATH"], "/usr/local/cuda-12.2/lib64/"
        )
        expected_calls = [
            call("CUDA_PATH", "/usr/local/cuda-12.2/", status=True),
            call("LD_LIBRARY_PATH", "/usr/local/cuda-12.2/lib64/", status=True),
        ]
        mock_log.assert_has_calls(expected_calls)

    @patch.dict(os.environ, {}, clear=True)
    def test_no_cuda_paths_found(self):
        module = cuda_module()
        with (
            patch.object(module, "get_os_name", return_value="linux"),
            patch.object(module, "log") as mock_log,
            patch.object(module, "paths", return_value=[]),
        ):
            module.set_cuda_env()
        self.assertNotIn("CUDA_PATH", os.environ)
        self.assertNotIn("LD_LIBRARY_PATH", os.environ)
        mock_log.assert_called_once_with(
            "Cuda not found", "Failed setting CUDA environment", status=False
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_only_cuda_path_found(self):
        module = cuda_module()
        with (
            patch.object(module, "get_os_name", return_value="linux"),
            patch.object(module, "log") as mock_log,
            patch.object(module, "paths") as mock_paths,
        ):
            mock_paths.side_effect = [["/opt/cuda/"], []]
            module.set_cuda_env()
        self.assertNotIn("CUDA_PATH", os.environ)
        self.assertNotIn("LD_LIBRARY_PATH", os.environ)
        mock_log.assert_called_once_with(
            "Cuda not found", "Failed setting CUDA environment", status=False
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_only_ld_library_path_found(self):
        module = cuda_module()
        with (
            patch.object(module, "get_os_name", return_value="linux"),
            patch.object(module, "log") as mock_log,
            patch.object(module, "paths") as mock_paths,
        ):
            mock_paths.side_effect = [[], ["/opt/cuda/lib/"]]
            module.set_cuda_env()
        self.assertNotIn("CUDA_PATH", os.environ)
        self.assertNotIn("LD_LIBRARY_PATH", os.environ)
        mock_log.assert_called_once_with(
            "Cuda not found", "Failed setting CUDA environment", status=False
        )


if __name__ == "__main__":
    unittest.main()
