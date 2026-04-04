import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from definers.application_data.loader_runtime import LoaderRuntimeSupport


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

        self.assertIn(self.file_path, result)
        self.assertIn(nested_path, result)
        mock_read.assert_called_once_with(self.test_dir)

    @patch("definers.system.read", side_effect=RuntimeError("boom"))
    def test_read_falls_back_to_file_contents(self, mock_read):
        result = LoaderRuntimeSupport.read(self.file_path)

        self.assertEqual(result, "payload")
        mock_read.assert_called_once_with(self.file_path)

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
