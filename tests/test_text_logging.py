import unittest
from unittest.mock import MagicMock, patch

import definers.os_utils as os_utils
import definers.path_utils as path_utils
import definers.text as text_module

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


class TestTextLogging(unittest.TestCase):
    @patch("requests.get")
    @patch("definers.logger.info", create=True)
    def test_google_translate_logs_result(self, mock_info, mock_get):
        mock_get.return_value = MagicMock(
            text='[[["hola","hello",null,null,10]]]'
        )

        res = text_module.google_translate("hello", lang="es")

        self.assertIsInstance(res, str)
        self.assertEqual(res, "hola")
        mock_info.assert_called_once_with(res)

    @patch("definers.logger.exception", create=True)
    def test_google_translate_handles_error(self, mock_exc):

        with patch("requests.get", side_effect=Exception("fail")):
            res = text_module.google_translate("hello", lang="es")
            self.assertEqual(res, "")
            mock_exc.assert_called_once()
