import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from definers.application_ml.training import initialize_linear_regression
from tests.torch_stubs import FakeModel, build_fake_torch


class TestInitializeLinearRegression(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(device=MagicMock(return_value="cpu"))
        self.logger = MagicMock()

    @patch("definers.system.secure_path", side_effect=lambda x: x)
    @patch("os.path.exists", return_value=False)
    def test_creates_new_model_if_not_exists(self, mock_exists, mock_sanitize):
        input_dim = 10
        model_path = "non_existent_model.pth"
        fake_torch = build_fake_torch()
        factory = MagicMock(return_value=FakeModel(input_dim))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            model = initialize_linear_regression(
                input_dim,
                model_path,
                runtime=self.runtime,
                factory=factory,
                logger=self.logger,
            )

        mock_exists.assert_called_once_with(model_path)
        factory.assert_called_once_with(input_dim)
        self.runtime.device.assert_called_once_with()
        self.assertEqual(str(next(model.parameters()).device), "cpu")
        self.assertIsNone(model.loaded_state_dict)
        self.logger.assert_called_once_with("Created new model.")
        self.assertIsInstance(model, FakeModel)

    @patch("definers.system.secure_path", side_effect=lambda x: x)
    @patch("os.path.exists", return_value=True)
    def test_loads_existing_model(self, mock_exists, mock_sanitize):
        input_dim = 5
        model_path = "existing_model.pth"
        mock_state_dict = {
            "linear.weight": [[0.0] * input_dim],
            "linear.bias": [0.0],
        }
        fake_torch = build_fake_torch(load_return_value=mock_state_dict)
        factory = MagicMock(return_value=FakeModel(input_dim))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            model = initialize_linear_regression(
                input_dim,
                model_path,
                runtime=self.runtime,
                factory=factory,
                logger=self.logger,
            )

        mock_exists.assert_called_once_with(model_path)
        factory.assert_called_once_with(input_dim)
        fake_torch.load.assert_called_once_with(
            model_path,
            map_location="cpu",
        )
        self.assertEqual(model.loaded_state_dict, mock_state_dict)
        self.assertEqual(str(next(model.parameters()).device), "cpu")
        self.logger.assert_called_once_with("Loaded existing model.")
        self.assertIsInstance(model, FakeModel)

    @patch("definers.system.secure_path", side_effect=ValueError("bad path"))
    @patch("os.path.exists")
    def test_returns_new_model_when_path_is_rejected(
        self,
        mock_exists,
        mock_sanitize,
    ):
        fake_torch = build_fake_torch()
        factory = MagicMock(return_value=FakeModel(4))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            model = initialize_linear_regression(
                4,
                "../blocked_model.pth",
                runtime=self.runtime,
                factory=factory,
                logger=self.logger,
            )

        factory.assert_called_once_with(4)
        mock_exists.assert_not_called()
        self.assertEqual(str(next(model.parameters()).device), "cpu")
        self.logger.assert_called_once_with(
            "Unsafe linear-regression model path: bad path"
        )
        self.assertIsInstance(model, FakeModel)


if __name__ == "__main__":
    unittest.main()
