import unittest
from types import SimpleNamespace

import definers.runtime_numpy as root_runtime_numpy
from definers.data.runtime_patches import (
    bootstrap_runtime_numpy,
    ensure_no_nep50_warning,
    get_array_module,
    get_numpy_module,
    init_cupy_numpy,
)


class TestRuntimePatches(unittest.TestCase):
    def test_runtime_patch_facade_matches_root_runtime_module(self):
        self.assertIs(init_cupy_numpy, root_runtime_numpy.init_cupy_numpy)
        self.assertIs(get_array_module, root_runtime_numpy.get_array_module)
        self.assertIs(get_numpy_module, root_runtime_numpy.get_numpy_module)

    def test_bootstrap_runtime_numpy_matches_accessor_helpers(self):
        np_module, numpy_module = bootstrap_runtime_numpy(force=True)

        self.assertIs(get_array_module(), np_module)
        self.assertIs(get_numpy_module(), numpy_module)

    def test_init_cupy_numpy_restores_removed_scalar_aliases(self):
        _, numpy_module = init_cupy_numpy()

        self.assertIs(numpy_module.float, numpy_module.float64)
        self.assertIs(numpy_module.int, numpy_module.int64)
        self.assertIs(numpy_module.bool, numpy_module.bool_)

    def test_init_cupy_numpy_preserves_recarray_api(self):
        _, numpy_module = init_cupy_numpy()

        self.assertTrue(hasattr(numpy_module, "rec"))
        self.assertTrue(hasattr(numpy_module.rec, "recarray"))
        self.assertIs(numpy_module.rec.recarray, numpy_module.recarray)

    def test_init_cupy_numpy_patches_finfo_once(self):
        _, numpy_module = init_cupy_numpy()
        patched_finfo = numpy_module.finfo

        _, numpy_module = init_cupy_numpy()

        self.assertIs(numpy_module.finfo, patched_finfo)

    def test_init_cupy_numpy_supports_integer_finfo_queries(self):
        _, numpy_module = init_cupy_numpy()

        finfo = numpy_module.finfo(numpy_module.int64)
        iinfo = numpy_module.iinfo(numpy_module.int64)

        self.assertEqual(finfo.min, iinfo.min)
        self.assertEqual(finfo.max, iinfo.max)

    def test_init_cupy_numpy_exposes_distutils_misc_util(self):
        _, numpy_module = init_cupy_numpy()

        self.assertTrue(hasattr(numpy_module, "distutils"))
        self.assertTrue(hasattr(numpy_module.distutils, "misc_util"))
        self.assertEqual(
            numpy_module.distutils.misc_util.get_info("blas_opt"),
            {},
        )

    def test_ensure_no_nep50_warning_behaves_like_decorator_factory(self):
        numpy_module = SimpleNamespace()
        sentinel = object()

        factory = ensure_no_nep50_warning(numpy_module)

        with factory():
            marker = sentinel

        def passthrough():
            return sentinel

        self.assertIs(marker, sentinel)
        self.assertIs(factory()(passthrough)(), sentinel)
        self.assertIs(factory, numpy_module._no_nep50_warning)

    def test_ensure_no_nep50_warning_preserves_existing_hook(self):
        _, numpy_module = init_cupy_numpy()
        existing = numpy_module._no_nep50_warning

        self.assertIs(ensure_no_nep50_warning(numpy_module), existing)


if __name__ == "__main__":
    unittest.main()
