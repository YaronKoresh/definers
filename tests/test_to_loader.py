import unittest
from unittest.mock import patch

from definers.application_data.preparation import to_loader
from tests.torch_stubs import build_fake_torch_modules


class TestToLoader(unittest.TestCase):
    def test_loader_properties(self):
        fake_torch, fake_torch_utils, fake_torch_data = (
            build_fake_torch_modules()
        )
        with patch.dict(
            "sys.modules",
            {
                "torch": fake_torch,
                "torch.utils": fake_torch_utils,
                "torch.utils.data": fake_torch_data,
            },
        ):
            X = fake_torch.randn(10, 5)
            y = fake_torch.randn(10)
            dataset = fake_torch_data.TensorDataset(X, y)
            batch_size = 2
            loader = to_loader(dataset, batch_size=batch_size)
            self.assertIsInstance(loader, fake_torch_data.DataLoader)
            self.assertEqual(loader.batch_size, batch_size)

    def test_default_batch_size(self):
        fake_torch, fake_torch_utils, fake_torch_data = (
            build_fake_torch_modules()
        )
        with patch.dict(
            "sys.modules",
            {
                "torch": fake_torch,
                "torch.utils": fake_torch_utils,
                "torch.utils.data": fake_torch_data,
            },
        ):
            X = fake_torch.randn(10, 5)
            dataset = fake_torch_data.TensorDataset(X)
            loader = to_loader(dataset)
            self.assertEqual(loader.batch_size, 1)

    def test_iteration(self):
        fake_torch, fake_torch_utils, fake_torch_data = (
            build_fake_torch_modules()
        )
        with patch.dict(
            "sys.modules",
            {
                "torch": fake_torch,
                "torch.utils": fake_torch_utils,
                "torch.utils.data": fake_torch_data,
            },
        ):
            X = fake_torch.randn(10, 5)
            dataset = fake_torch_data.TensorDataset(X)
            loader = to_loader(dataset, batch_size=4)
            batches = list(loader)
            self.assertEqual(len(batches), 3)
            self.assertEqual(batches[0][0].shape, (4, 5))
            self.assertEqual(batches[1][0].shape, (4, 5))
            self.assertEqual(batches[2][0].shape, (2, 5))


if __name__ == "__main__":
    unittest.main()
