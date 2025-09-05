import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np

class DummyModel(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
        self.parameters_called = False

    def forward(self, x):
        return self.linear(x)
        
    def parameters(self):
        self.parameters_called = True
        return super().parameters()

    def zero_grad(self):
        pass

    def step(self):
        pass

    def backward(self):
        pass

patch_target_init = 'definers.initialize_linear_regression'

from definers import train_linear_regression

class TestTrainLinearRegression(unittest.TestCase):

    @patch('torch.save')
    @patch(patch_target_init)
    def test_training_step_execution(self, mock_initialize, mock_torch_save):
        input_dim = 3
        model_path = "test_model.pth"
        
        mock_model_instance = DummyModel(input_dim)
        mock_initialize.return_value = mock_model_instance
        
        mock_optimizer_instance = MagicMock()
        mock_optimizer_instance.zero_grad = MagicMock()
        mock_optimizer_instance.step = MagicMock()
        
        with patch('torch.optim.SGD', return_value=mock_optimizer_instance) as mock_sgd:
            
            X = np.random.rand(10, input_dim).astype(np.float32)
            y = np.random.rand(10).astype(np.float32)
            learning_rate = 0.05

            trained_model = train_linear_regression(X, y, model_path, learning_rate=learning_rate)
            
            mock_initialize.assert_called_once_with(input_dim, model_path)
            
            self.assertTrue(mock_model_instance.parameters_called)
            mock_sgd.assert_called_once()
            self.assertEqual(mock_sgd.call_args.kwargs['lr'], learning_rate)

            mock_optimizer_instance.zero_grad.assert_called_once()
            mock_optimizer_instance.step.assert_called_once()
            
            mock_torch_save.assert_called_once()
            self.assertEqual(mock_torch_save.call_args[0][0], mock_model_instance.state_dict())
            self.assertEqual(mock_torch_save.call_args[0][1], model_path)
            
            self.assertIs(trained_model, mock_model_instance)

    @patch('torch.save')
    @patch(patch_target_init)
    def test_data_conversion_to_tensors(self, mock_initialize, mock_torch_save):
        input_dim = 2
        model_path = "tensor_test.pth"
        
        mock_model_instance = MagicMock(return_value=torch.randn(5, 1))
        mock_initialize.return_value = mock_model_instance
        
        X_np = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.float32)
        y_np = np.array([3, 7, 11, 15, 19], dtype=np.float32)
        
        with patch('torch.optim.SGD'):
            train_linear_regression(X_np, y_np, model_path)
        
        self.assertEqual(mock_model_instance.call_count, 1)
        args, kwargs = mock_model_instance.call_args
        
        X_arg = args[0]
        self.assertIsInstance(X_arg, torch.Tensor)
        self.assertEqual(X_arg.device.type, 'cuda')
        self.assertEqual(X_arg.dtype, torch.float32)
        np.testing.assert_array_equal(X_arg.cpu().numpy(), X_np)

if __name__ == '__main__':
    unittest.main()
