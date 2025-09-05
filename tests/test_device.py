import unittest
from unittest.mock import patch, MagicMock
from definers import device

class TestDevice(unittest.TestCase):

    @patch('definers.Accelerator')
    def test_device_returns_cuda_when_available(self, mock_accelerator):
        mock_instance = MagicMock()
        mock_instance.device = 'cuda'
        mock_accelerator.return_value = mock_instance

        self.assertEqual(device(), 'cuda')
        mock_accelerator.assert_called_once()

    @patch('definers.Accelerator')
    def test_device_returns_cpu_when_cuda_not_available(self, mock_accelerator):
        mock_instance = MagicMock()
        mock_instance.device = 'cpu'
        mock_accelerator.return_value = mock_instance

        self.assertEqual(device(), 'cpu')
        mock_accelerator.assert_called_once()

if __name__ == '__main__':
    unittest.main()
