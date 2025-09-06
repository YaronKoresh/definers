import unittest
import sys
from unittest.mock import patch, MagicMock

# Temporarily remove cupy from sys.modules to test fallback
if 'cupy' in sys.modules:
    del sys.modules['cupy']

from definers import _init_cupy_numpy

class TestInitCupyNumpy(unittest.TestCase):

    @patch.dict('sys.modules', {'cupy': None})
    def test_fallback_to_numpy_when_cupy_not_installed(self):
        import numpy
        np, _np = _init_cupy_numpy()
        self.assertIs(np, numpy)
        self.assertIs(_np, numpy)

    @patch('importlib.util.find_spec')
    def test_uses_cupy_when_installed(self, mock_find_spec):
        # We can't actually install cupy, so we mock its presence
        mock_find_spec.return_value = True
        mock_cupy = MagicMock()
        mock_cupy.float64 = "float64_val"
        mock_cupy.int64 = "int64_val"
        with patch.dict('sys.modules', {'cupy': mock_cupy}):
            np, _np = _init_cupy_numpy()
            self.assertIs(np, mock_cupy)

    def test_float_attribute_is_set_on_numpy(self):
        import numpy
        np, _ = _init_cupy_numpy()
        self.assertTrue(hasattr(np, 'float'))
        self.assertEqual(np.float, np.float64)

    def test_int_attribute_is_set_on_numpy(self):
        import numpy
        np, _ = _init_cupy_numpy()
        self.assertTrue(hasattr(np, 'int'))
        self.assertEqual(np.int, np.int64)

    @patch('importlib.util.find_spec')
    def test_float_attribute_is_set_on_cupy(self, mock_find_spec):
        mock_find_spec.return_value = True
        mock_cupy = MagicMock()
        mock_cupy.float64 = "float64_val"
        # Simulate that 'float' is not present
        del mock_cupy.float
        with patch.dict('sys.modules', {'cupy': mock_cupy}):
             np, _ = _init_cupy_numpy()
             self.assertTrue(hasattr(np, 'float'))
             self.assertEqual(np.float, "float64_val")

    @patch('importlib.util.find_spec')
    def test_int_attribute_is_set_on_cupy(self, mock_find_spec):
        mock_find_spec.return_value = True
        mock_cupy = MagicMock()
        mock_cupy.int64 = "int64_val"
        # Simulate that 'int' is not present
        del mock_cupy.int
        with patch.dict('sys.modules', {'cupy': mock_cupy}):
            np, _ = _init_cupy_numpy()
            self.assertTrue(hasattr(np, 'int'))
            self.assertEqual(np.int, "int64_val")

    def test_returns_two_modules(self):
        np, _np = _init_cupy_numpy()
        self.assertTrue(hasattr(np, '__version__'))
        self.assertTrue(hasattr(_np, '__version__'))

if __name__ == '__main__':
    unittest.main()
