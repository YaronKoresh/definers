import os
import re
import unittest
from unittest.mock import MagicMock

import definers
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

from definers.image import save_image


class TestSaveImage(unittest.TestCase):
    def test_save_image_returns_png_path(self):
        mock_img = MagicMock()
        with unittest.mock.patch.object(
            definers, "random_string", create=True, return_value="random123"
        ):
            result_path = save_image(mock_img, path=".")
        self.assertEqual(os.path.dirname(result_path), ".")
        self.assertEqual(os.path.basename(result_path), "img_random123.png")
        mock_img.save.assert_called_once_with(result_path)

    def test_save_image_calls_save_method(self):
        mock_img = MagicMock()
        with unittest.mock.patch.object(
            definers,
            "random_string",
            create=True,
            return_value="another_random",
        ):
            result_path = save_image(mock_img, path="/tmp")
        self.assertEqual(os.path.dirname(result_path), "/tmp")
        self.assertEqual(
            os.path.basename(result_path), "img_another_random.png"
        )
        mock_img.save.assert_called_once_with(result_path)


if __name__ == "__main__":
    unittest.main()
