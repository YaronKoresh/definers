import unittest
from unittest.mock import patch
import os
import sys
from definers import pre_install

class TestPreInstall(unittest.TestCase):

    @patch.dict(os.environ, {}, clear=True)
    def test_pre_install_sets_environment_variables(self):
        pre_install()

        self.assertEqual(os.environ['TRANSFORMERS_CACHE'], '/opt/ml/checkpoints/')
        self.assertEqual(os.environ['HF_DATASETS_CACHE'], '/opt/ml/checkpoints/')
        self.assertEqual(os.environ["GRADIO_ALLOW_FLAGGING"], "never")
        self.assertEqual(os.environ["OMP_NUM_THREADS"], "4")
        self.assertEqual(os.environ["DISPLAY"], ":0.0")
        self.assertEqual(os.environ["NUMBA_CACHE_DIR"], f'{os.environ["HOME"]}/.tmp')
        self.assertEqual(os.environ["DISABLE_FLASH_ATTENTION"], "True")

        if sys.platform == "darwin":
            self.assertEqual(os.environ["PYTORCH_ENABLE_MPS_FALLBACK"], "1")
        else:
            self.assertNotIn("PYTORCH_ENABLE_MPS_FALLBACK", os.environ)

if __name__ == '__main__':
    unittest.main()
