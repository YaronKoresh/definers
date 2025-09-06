import unittest

import torch

from definers import tensor_length

class TestTensorLength(unittest.TestCase):

    def test_scalar_tensor(self):
        tensor = torch.tensor(5)
        self.assertEqual(tensor_length(tensor), 1)

    def test_1d_tensor(self):
        tensor = torch.tensor([1, 2, 3, 4])
        self.assertEqual(tensor_length(tensor), 4)

    def test_2d_tensor(self):
        tensor = torch.tensor([[1, 2], [3, 4]])
        self.assertEqual(tensor_length(tensor), 4)

    def test_3d_tensor(self):
        tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        self.assertEqual(tensor_length(tensor), 8)

    def test_empty_tensor(self):
        tensor = torch.tensor([])
        self.assertEqual(tensor_length(tensor), 0)

    def test_large_tensor(self):
        tensor = torch.randn(10, 20, 5)
        self.assertEqual(tensor_length(tensor), 1000)


if __name__ == "__main__":
    unittest.main()
