import unittest
import torch
from torch.utils.data import TensorDataset, DataLoader
from definers import to_loader

class TestToLoader(unittest.TestCase):

    def setUp(self):
        self.features = torch.randn(10, 5)
        self.labels = torch.randn(10, 1)
        self.dataset = TensorDataset(self.features, self.labels)

    def test_returns_dataloader_instance(self):
        loader = to_loader(self.dataset)
        self.assertIsInstance(loader, DataLoader)

    def test_default_batch_size(self):
        loader = to_loader(self.dataset)
        self.assertEqual(loader.batch_size, 1)

    def test_custom_batch_size(self):
        loader = to_loader(self.dataset, batch_size=4)
        self.assertEqual(loader.batch_size, 4)

    def test_loader_properties(self):
        loader = to_loader(self.dataset)
        self.assertTrue(loader.shuffle)
        self.assertFalse(loader.pin_memory)
        self.assertEqual(loader.num_workers, 0)
        self.assertFalse(loader.drop_last)

    def test_iteration_over_loader(self):
        loader = to_loader(self.dataset, batch_size=2)
        batch_count = 0
        total_samples = 0
        for batch in loader:
            batch_count += 1
            self.assertEqual(len(batch), 2)
            self.assertEqual(batch[0].shape[0], 2)
            self.assertEqual(batch[1].shape[0], 2)
            total_samples += batch[0].shape[0]
        
        self.assertEqual(batch_count, 5)
        self.assertEqual(total_samples, 10)

if __name__ == '__main__':
    unittest.main()
