import sys
import unittest
from unittest.mock import MagicMock, patch

from definers import post_install


class TestPostInstall(unittest.TestCase):

    @patch("definers.free")
    @patch(
        "torch.fx.experimental.proxy_tensor.get_proxy_mode",
        create=True,
    )
    def test_post_install_calls_free(
        self, mock_get_proxy_mode, mock_free
    ):
        post_install()
        mock_free.assert_called_once()

    @patch("definers.free")
    @patch("torch.fx.experimental.proxy_tensor")
    def test_post_install_monkeypatches_proxy_tensor(
        self, mock_proxy_tensor, mock_free
    ):
        if hasattr(mock_proxy_tensor, "get_proxy_mode"):
            delattr(mock_proxy_tensor, "get_proxy_mode")

        post_install()

        self.assertTrue(hasattr(mock_proxy_tensor, "get_proxy_mode"))
        self.assertTrue(
            callable(getattr(mock_proxy_tensor, "get_proxy_mode"))
        )

    @patch("definers.free")
    @patch(
        "numpy._no_nep50_warning", create=True, new_callable=MagicMock
    )
    def test_post_install_monkeypatches_numpy(
        self, mock_np_warn, mock_free
    ):
        if hasattr(sys.modules["numpy"], "_no_nep50_warning"):
            pass

        post_install()
        self.assertTrue(
            hasattr(sys.modules["numpy"], "_no_nep50_warning")
        )
        self.assertTrue(
            callable(
                getattr(sys.modules["numpy"], "_no_nep50_warning")
            )
        )


if __name__ == "__main__":
    import numpy
    import torch

    unittest.main()
