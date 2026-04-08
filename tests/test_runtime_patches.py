import unittest

from definers.application_data.runtime_patches import init_cupy_numpy


class TestRuntimePatches(unittest.TestCase):
    def test_init_cupy_numpy_preserves_recarray_api(self):
        _, numpy_module = init_cupy_numpy()

        self.assertTrue(hasattr(numpy_module, "rec"))
        self.assertTrue(hasattr(numpy_module.rec, "recarray"))
        self.assertIs(numpy_module.rec.recarray, numpy_module.recarray)


if __name__ == "__main__":
    unittest.main()
