import unittest
from unittest.mock import patch

from definers.data.arrays import dtype
from tests.torch_stubs import build_fake_torch


class TestDtype(unittest.TestCase):
    def test_bf16_supported(self):
        fake_torch = build_fake_torch()

        with patch.dict("sys.modules", {"torch": fake_torch}):
            with patch.object(
                fake_torch.cuda,
                "is_bf16_supported",
                return_value=True,
            ):
                self.assertEqual(dtype(), fake_torch.bfloat16)

    def test_bf16_not_supported(self):
        fake_torch = build_fake_torch()

        with patch.dict("sys.modules", {"torch": fake_torch}):
            with patch.object(
                fake_torch.cuda,
                "is_bf16_supported",
                return_value=False,
            ):
                self.assertEqual(dtype(), fake_torch.float16)


if __name__ == "__main__":
    unittest.main()
