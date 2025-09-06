import unittest
import torch
from definers import tensor_length

class TestTensorLength(unittest.TestCase):

    def test_scalar_tensor(self):
        tensor = torch.tensor(5)
        self.assertEqual(tensor_length(tensor), 1)

    def test_1d_tensor(self):
        tensor = torch.randn(10)
        self.assertEqual(tensor_length(tensor), 10)

    def test_2d_tensor(self):
        tensor = torch.randn(3, 4)
        self.assertEqual(tensor_length(tensor), 12)

    def test_3d_tensor(self):
        tensor = torch.randn(2, 3, 5)
        self.assertEqual(tensor_length(tensor), 30)

    def test_empty_tensor(self):
        tensor = torch.randn(5, 0, 5)
        self.assertEqual(tensor_length(tensor), 0)

    def test_large_tensor(self):
        tensor = torch.randn(100, 100)
        self.assertEqual(tensor_length(tensor), 10000)

if __name__ == '__main__':
    unittest.main()
