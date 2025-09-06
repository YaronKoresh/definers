import unittest

import torch
from torch.utils.data import TensorDataset

from definers import to_loader


class TestToLoader(unittest.TestCase):

    def test_loader_properties(self):
        X = torch.randn(10, 5)
        y = torch.randn(10)
        dataset = TensorDataset(X, y)
        batch_size = 2

        loader = to_loader(dataset, batch_size=batch_size)

        from torch.utils.data import DataLoader

        self.assertIsInstance(loader, DataLoader)
        self.assertEqual(loader.batch_size, batch_size)

    def test_default_batch_size(self):
        X = torch.randn(10, 5)
        dataset = TensorDataset(X)
        loader = to_loader(dataset)
        self.assertEqual(loader.batch_size, 1)

    def test_iteration(self):
        X = torch.randn(10, 5)
        dataset = TensorDataset(X)
        loader = to_loader(dataset, batch_size=4)

        batches = list(loader)
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0][0].shape, (4, 5))
        self.assertEqual(batches[1][0].shape, (4, 5))
        self.assertEqual(batches[2][0].shape, (2, 5))


if __name__ == "__main__":
    unittest.main()
