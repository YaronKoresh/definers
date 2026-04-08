import sys
import types
import unittest
from unittest.mock import patch

from definers.system import post_install


class TestPostInstall(unittest.TestCase):
    @patch("definers.cuda.free")
    def test_post_install_calls_cuda_free(self, mock_free):
        post_install()
        mock_free.assert_called_once()

    @patch("definers.cuda.free")
    def test_post_install_monkeypatches_proxy_tensor(self, mock_free):
        proxy_tensor_module = types.ModuleType("proxy_tensor")
        experimental_module = types.ModuleType("experimental")
        fx_module = types.ModuleType("fx")
        torch_module = types.ModuleType("torch")
        experimental_module.proxy_tensor = proxy_tensor_module
        fx_module.experimental = experimental_module
        torch_module.fx = fx_module

        with patch.dict(
            sys.modules,
            {
                "torch": torch_module,
                "torch.fx": fx_module,
                "torch.fx.experimental": experimental_module,
                "torch.fx.experimental.proxy_tensor": proxy_tensor_module,
            },
        ):
            post_install()

        self.assertTrue(hasattr(proxy_tensor_module, "get_proxy_mode"))
        self.assertTrue(
            callable(getattr(proxy_tensor_module, "get_proxy_mode"))
        )
        mock_free.assert_called_once()

    @patch("definers.cuda.free")
    def test_post_install_monkeypatches_numpy(self, mock_free):
        import numpy

        had_original = hasattr(numpy, "_no_nep50_warning")
        original = getattr(numpy, "_no_nep50_warning", None)
        if had_original:
            delattr(numpy, "_no_nep50_warning")
        try:
            post_install()
            self.assertTrue(hasattr(numpy, "_no_nep50_warning"))
            self.assertTrue(callable(getattr(numpy, "_no_nep50_warning")))
            mock_free.assert_called_once()
        finally:
            if had_original:
                numpy._no_nep50_warning = original
            elif hasattr(numpy, "_no_nep50_warning"):
                delattr(numpy, "_no_nep50_warning")


if __name__ == "__main__":
    import numpy

    unittest.main()
