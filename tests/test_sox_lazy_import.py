import contextlib
import importlib
import io
import sys
import unittest
from unittest import mock


class TestSoxLazyImport(unittest.TestCase):
    def setUp(self):

        self._orig_import = importlib.import_module

        def fake_import(name, package=None):
            if name == "sox":
                raise ImportError
            return self._orig_import(name, package)

        self._patcher = mock.patch(
            "importlib.import_module", side_effect=fake_import
        )
        self._patcher.start()

    def tearDown(self):
        if hasattr(self, "_patcher"):
            self._patcher.stop()
        for name in ("sox", "definers"):
            if name in sys.modules:
                del sys.modules[name]

    def test_import_with_missing_sox_module(self):
        if "definers" in sys.modules:
            del sys.modules["definers"]

        original_import = importlib.import_module

        def fake_import(name, package=None):
            if name == "sox":
                raise ImportError("Mocked SoX missing")
            return original_import(name, package)

        with mock.patch("importlib.import_module", side_effect=fake_import):
            buf = io.StringIO()
            with (
                contextlib.redirect_stderr(buf),
                contextlib.redirect_stdout(buf),
            ):
                import definers

                importlib.reload(definers)
            out = buf.getvalue()

        self.assertNotIn("sox is not recognized", out.lower())
        self.assertFalse(
            definers.has_sox(), "has_sox() should be False when import fails"
        )

        if hasattr(definers, "sox"):
            try:
                definers.sox.Transformer()
                self.fail(
                    "Should have raised ImportError when calling Transformer without sox installed"
                )
            except (ImportError, AttributeError):
                pass

    def test_import_again_after_failure(self):

        import definers

        sys.modules["sox"] = mock.MagicMock()
        importlib.reload(definers)

        self.assertTrue(hasattr(definers.sox, "Transformer"))
