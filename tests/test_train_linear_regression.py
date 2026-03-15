import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from definers.application_ml.training import (
    LinearRegressionTorch,
    train_linear_regression,
)


class TestTrainLinearRegression(unittest.TestCase):
    def setUp(self):
        self.input_dim = 3
        self.X = np.random.rand(10, self.input_dim).astype(np.float32)
        self.y = np.random.rand(10).astype(np.float32)
        self.model_path = "test_model.pth"

    def tearDown(self):
        import os

        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    @patch("torch.save")
    def test_data_conversion_to_tensors(self, mock_torch_save):
        runtime = MagicMock()
        runtime.device.return_value = "cpu"
        runtime.initialize_linear_regression.return_value = (
            LinearRegressionTorch(self.input_dim)
        )

        model = train_linear_regression(
            self.X,
            self.y,
            self.model_path,
            runtime=runtime,
        )

        runtime.initialize_linear_regression.assert_called_once_with(
            self.input_dim,
            self.model_path,
        )
        self.assertEqual(str(next(model.parameters()).device), "cpu")

    @patch("torch.optim.SGD")
    @patch("torch.nn.MSELoss")
    @patch("torch.save")
    def test_training_step_execution(
        self, mock_torch_save, mock_mse_loss, mock_sgd
    ):
        runtime = MagicMock()
        runtime.device.return_value = "cpu"
        runtime.initialize_linear_regression.return_value = (
            LinearRegressionTorch(self.input_dim)
        )
        mock_optimizer_instance = MagicMock()
        mock_sgd.return_value = mock_optimizer_instance
        mock_loss_instance = MagicMock()
        mock_loss_instance.return_value = torch.tensor(0.5, requires_grad=True)
        mock_mse_loss.return_value = mock_loss_instance
        train_linear_regression(
            self.X,
            self.y,
            self.model_path,
            runtime=runtime,
        )
        mock_optimizer_instance.zero_grad.assert_called_once()
        mock_loss_instance.return_value.backward.assert_called_once()
        mock_optimizer_instance.step.assert_called_once()
        mock_torch_save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
