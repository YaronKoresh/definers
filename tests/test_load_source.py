import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from definers.application_data.loaders import load_source


class TestLoadSource(unittest.TestCase):
    @patch("definers.application_data.loaders._runtime")
    def test_load_source_uses_runtime_fetch_dataset(self, mock_runtime):
        runtime = SimpleNamespace(
            fetch_dataset=MagicMock(return_value="remote-dataset")
        )
        mock_runtime.return_value = runtime

        result = load_source(
            remote_src="dataset/name", url_type="json", revision="v1"
        )

        self.assertEqual(result, "remote-dataset")
        runtime.fetch_dataset.assert_called_once_with(
            "dataset/name", "json", "v1"
        )

    @patch("definers.application_data.loaders._runtime")
    def test_load_source_uses_runtime_files_to_dataset(self, mock_runtime):
        runtime = SimpleNamespace(
            files_to_dataset=MagicMock(return_value="tensor-dataset")
        )
        mock_runtime.return_value = runtime

        result = load_source(features=["feature.csv"], labels=["label.csv"])

        self.assertEqual(result, "tensor-dataset")
        runtime.files_to_dataset.assert_called_once_with(
            ["feature.csv"], ["label.csv"]
        )

    def test_load_source_returns_none_without_inputs(self):
        self.assertIsNone(load_source())


if __name__ == "__main__":
    unittest.main()
