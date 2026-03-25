import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from definers.application_ml.training import initialize_linear_regression


class DummyModel(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

    def cuda(self):
        pass

    def load_state_dict(self, state_dict):
        pass


class TestInitializeLinearRegression(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(device=MagicMock(return_value="cpu"))
        self.logger = MagicMock()

    @patch("definers.system.secure_path", side_effect=lambda x: x)
    @patch("os.path.exists", return_value=False)
    def test_creates_new_model_if_not_exists(self, mock_exists, mock_sanitize):
        input_dim = 10
        model_path = "non_existent_model.pth"
        factory = MagicMock(return_value=DummyModel(input_dim))
        mock_model_instance = factory.return_value
        mock_model_instance.to = MagicMock()
        mock_model_instance.load_state_dict = MagicMock()
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
        mock_model_instance.to.assert_called_once_with("cpu")
        mock_model_instance.load_state_dict.assert_not_called()
        self.logger.assert_called_once_with("Created new model.")
        self.assertIsInstance(model, DummyModel)

    @patch("definers.system.secure_path", side_effect=lambda x: x)
    @patch("os.path.exists", return_value=True)
    @patch("torch.load")
    def test_loads_existing_model(
        self, mock_torch_load, mock_exists, mock_sanitize
    ):
        input_dim = 5
        model_path = "existing_model.pth"
        factory = MagicMock(return_value=DummyModel(input_dim))
        mock_state_dict = {
            "linear.weight": torch.randn(1, 5),
            "linear.bias": torch.randn(1),
        }
        mock_torch_load.return_value = mock_state_dict
        mock_model_instance = factory.return_value
        mock_model_instance.to = MagicMock()
        mock_model_instance.load_state_dict = MagicMock()
        model = initialize_linear_regression(
            input_dim,
            model_path,
            runtime=self.runtime,
            factory=factory,
            logger=self.logger,
        )

        mock_exists.assert_called_once_with(model_path)
        factory.assert_called_once_with(input_dim)
        mock_torch_load.assert_called_once_with(
            model_path,
            map_location="cpu",
        )
        mock_model_instance.load_state_dict.assert_called_once_with(
            mock_state_dict
        )
        mock_model_instance.to.assert_called_once_with("cpu")
        self.logger.assert_called_once_with("Loaded existing model.")
        self.assertIsInstance(model, DummyModel)

    @patch("definers.system.secure_path", side_effect=ValueError("bad path"))
    @patch("os.path.exists")
    def test_returns_new_model_when_path_is_rejected(
        self,
        mock_exists,
        mock_sanitize,
    ):
        factory = MagicMock(return_value=DummyModel(4))
        mock_model_instance = factory.return_value
        mock_model_instance.to = MagicMock()
        model = initialize_linear_regression(
            4,
            "../blocked_model.pth",
            runtime=self.runtime,
            factory=factory,
            logger=self.logger,
        )

        factory.assert_called_once_with(4)
        mock_exists.assert_not_called()
        mock_model_instance.to.assert_called_once_with("cpu")
        self.logger.assert_called_once_with(
            "Unsafe linear-regression model path: bad path"
        )
        self.assertIsInstance(model, DummyModel)


if __name__ == "__main__":
    unittest.main()
