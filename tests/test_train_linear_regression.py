import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from definers.application_ml.training import train_linear_regression
from tests.torch_stubs import FakeModel, build_fake_torch


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

    def test_data_conversion_to_tensors(self):
        fake_torch = build_fake_torch()
        runtime = MagicMock()
        runtime.device.return_value = "cpu"
        runtime.initialize_linear_regression.return_value = FakeModel(
            self.input_dim
        )

        with patch.dict(sys.modules, {"torch": fake_torch}):
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
        fake_torch.save.assert_called_once()

    def test_training_step_execution(self):
        fake_torch = build_fake_torch()
        runtime = MagicMock()
        runtime.device.return_value = "cpu"
        runtime.initialize_linear_regression.return_value = FakeModel(
            self.input_dim
        )

        with patch.dict(sys.modules, {"torch": fake_torch}):
            train_linear_regression(
                self.X,
                self.y,
                self.model_path,
                runtime=runtime,
            )

        fake_torch.optim.SGD.return_value.zero_grad.assert_called_once()
        fake_torch.nn.MSELoss.return_value.return_value.backward.assert_called_once()
        fake_torch.optim.SGD.return_value.step.assert_called_once()
        fake_torch.save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
