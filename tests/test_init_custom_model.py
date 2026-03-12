import unittest
from unittest.mock import MagicMock, mock_open, patch

from definers import init_custom_model


class TestInitCustomModel(unittest.TestCase):
    @patch("definers.onnx")
    @patch("definers.pickle")
    def test_init_custom_model_invalid_type(self, mock_pickle, mock_onnx):
        model = init_custom_model("invalid_type", "dummy_path")
        self.assertIsNone(model)

    @patch("definers.onnx")
    @patch("definers.pickle")
    def test_init_custom_model_no_path(self, mock_pickle, mock_onnx):
        model_onnx = init_custom_model("onnx", None)
        self.assertIsNone(model_onnx)
        model_pkl = init_custom_model("pkl", None)
        self.assertIsNone(model_pkl)

    @patch("definers._system.secure_path", side_effect=lambda x: x)
    @patch("builtins.open", new_callable=mock_open)
    @patch("definers.onnx")
    def test_init_custom_model_onnx_load_fails(
        self, mock_onnx, mock_file, mock_sanitize
    ):
        mock_onnx.load.side_effect = Exception("ONNX load error")
        model = init_custom_model("onnx", "model.onnx")
        self.assertIsNone(model)

    @patch("definers._system.secure_path", side_effect=lambda x: x)
    @patch("builtins.open", new_callable=mock_open)
    @patch("definers.pickle")
    def test_init_custom_model_pkl_load_fails(
        self, mock_pickle, mock_file, mock_sanitize
    ):
        mock_pickle.load.side_effect = Exception("Pickle load error")
        model = init_custom_model("pkl", "model.pkl")
        self.assertIsNone(model)

    @patch("definers._system.secure_path", side_effect=lambda x: x)
    @patch("definers.catch")
    @patch("definers.onnx")
    def test_init_custom_model_general_exception(
        self, mock_onnx, mock_catch, mock_sanitize
    ):
        mock_onnx.load.side_effect = TypeError("Some internal error")
        model = init_custom_model("onnx", "model.onnx")
        self.assertIsNone(model)


if __name__ == "__main__":
    unittest.main()
