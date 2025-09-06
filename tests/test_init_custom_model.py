import unittest
from unittest.mock import MagicMock, mock_open, patch

from definers import init_custom_model


class TestInitCustomModel(unittest.TestCase):

    @patch("definers.onnx")
    @patch("definers.pickle")
    def test_init_custom_model_invalid_type(
        self, mock_pickle, mock_onnx
    ):
        model = init_custom_model("invalid_type", "dummy_path")
        self.assertIsNone(model)

    @patch("definers.onnx")
    @patch("definers.pickle")
    def test_init_custom_model_no_path(self, mock_pickle, mock_onnx):
        model_onnx = init_custom_model("onnx", None)
        self.assertIsNone(model_onnx)
        model_pkl = init_custom_model("pkl", None)
        self.assertIsNone(model_pkl)

    @patch("builtins.open", new_callable=mock_open)
    @patch("definers.onnx")
    def test_init_custom_model_onnx_success(
        self, mock_onnx, mock_file
    ):
        mock_model = MagicMock()
        mock_onnx.load.return_value = mock_model

        model = init_custom_model("onnx", "model.onnx")

        mock_file.assert_called_once_with("model.onnx", "rb")
        mock_onnx.load.assert_called_once()
        self.assertEqual(model, mock_model)

    @patch("builtins.open", new_callable=mock_open)
    @patch("definers.pickle")
    def test_init_custom_model_pkl_success(
        self, mock_pickle, mock_file
    ):
        mock_model = MagicMock()
        mock_pickle.load.return_value = mock_model

        model = init_custom_model("pkl", "model.pkl")

        mock_file.assert_called_once_with("model.pkl", "rb")
        mock_pickle.load.assert_called_once()
        self.assertEqual(model, mock_model)

    @patch("builtins.open", new_callable=mock_open)
    @patch("definers.onnx")
    def test_init_custom_model_onnx_load_fails(
        self, mock_onnx, mock_file
    ):
        mock_onnx.load.side_effect = Exception("ONNX load error")

        model = init_custom_model("onnx", "model.onnx")

        self.assertIsNone(model)

    @patch("builtins.open", new_callable=mock_open)
    @patch("definers.pickle")
    def test_init_custom_model_pkl_load_fails(
        self, mock_pickle, mock_file
    ):
        mock_pickle.load.side_effect = Exception("Pickle load error")

        model = init_custom_model("pkl", "model.pkl")

        self.assertIsNone(model)

    @patch("definers.catch")
    @patch("definers.onnx")
    def test_init_custom_model_general_exception(
        self, mock_onnx, mock_catch
    ):
        mock_onnx.load.side_effect = TypeError("Some internal error")

        model = init_custom_model("onnx", "model.onnx")

        self.assertIsNone(model)
        mock_catch.assert_called_once()
        self.assertIn(
            "Error initializing model", mock_catch.call_args[0][0]
        )


if __name__ == "__main__":
    unittest.main()
