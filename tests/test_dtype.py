import unittest
from unittest.mock import patch

import torch

from definers import dtype


class TestDtype(unittest.TestCase):

    @patch("torch.cuda.is_bf16_supported", return_value=True)
    def test_bf16_supported(self, mock_is_bf16_supported):
        self.assertEqual(dtype(), torch.bfloat16)

    @patch("torch.cuda.is_bf16_supported", return_value=False)
    def test_bf16_not_supported(self, mock_is_bf16_supported):
        self.assertEqual(dtype(), torch.float16)


if __name__ == "__main__":
    unittest.main()
