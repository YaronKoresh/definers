import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from definers.data.loader_runtime import LoaderRuntimeSupport


def _normalized_path(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


class TestLoaderRuntimeSupport(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.file_path = os.path.join(self.test_dir, "sample.txt")
        with open(self.file_path, "w", encoding="utf-8") as handle:
            handle.write("payload")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("definers.system.read", side_effect=RuntimeError("boom"))
    def test_read_falls_back_to_directory_listing(self, mock_read):
        nested_path = os.path.join(self.test_dir, "nested.txt")
        with open(nested_path, "w", encoding="utf-8") as handle:
            handle.write("nested")

        result = LoaderRuntimeSupport.read(self.test_dir)

        normalized_result = {_normalized_path(path) for path in result}
        self.assertIn(_normalized_path(self.file_path), normalized_result)
        self.assertIn(_normalized_path(nested_path), normalized_result)
        mock_read.assert_called_once()
        self.assertEqual(
            _normalized_path(mock_read.call_args.args[0]),
            _normalized_path(self.test_dir),
        )

    @patch("definers.system.read", side_effect=RuntimeError("boom"))
    def test_read_falls_back_to_file_contents(self, mock_read):
        result = LoaderRuntimeSupport.read(self.file_path)

        self.assertEqual(result, "payload")
        mock_read.assert_called_once()
        self.assertEqual(
            _normalized_path(mock_read.call_args.args[0]),
            _normalized_path(self.file_path),
        )

    @patch(
        "definers.system.secure_path", side_effect=lambda path, trust=None: path
    )
    def test_read_validates_path_with_secure_path(self, mock_secure_path):
        LoaderRuntimeSupport.read(self.file_path)

        self.assertEqual(mock_secure_path.call_count, 1)
        self.assertEqual(mock_secure_path.call_args.args[0], self.file_path)
        self.assertIn(
            _normalized_path(tempfile.gettempdir()),
            {
                _normalized_path(path)
                for path in mock_secure_path.call_args.kwargs["trust"]
            },
        )

    @patch("definers.system.tmp", side_effect=RuntimeError("boom"))
    def test_tmp_falls_back_to_local_tempfile(self, mock_tmp):
        path = LoaderRuntimeSupport.tmp("json")

        try:
            self.assertTrue(path.endswith(".json"))
            self.assertTrue(os.path.exists(path))
        finally:
            if os.path.exists(path):
                os.unlink(path)
        mock_tmp.assert_called_once_with("json")


if __name__ == "__main__":
    unittest.main()
