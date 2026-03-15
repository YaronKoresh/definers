import contextlib
import importlib
import io
import sys
import unittest
from unittest import mock


class TestSoxLazyImport(unittest.TestCase):
    def tearDown(self):
        for name in ("sox", "definers"):
            if name in sys.modules:
                del sys.modules[name]

    def test_import_with_missing_sox_module(self):
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
        original_import = importlib.import_module

        def fake_import(name, package=None):
            if name == "sox":
                raise ImportError("Mocked SoX missing")
            return original_import(name, package)

        with mock.patch("importlib.import_module", side_effect=fake_import):
            import definers

            importlib.reload(definers)

        self.assertFalse(definers.has_sox())

        cached_sox = mock.MagicMock()
        sys.modules["sox"] = cached_sox

        with mock.patch(
            "importlib.import_module", side_effect=AssertionError
        ) as mocked_import:
            importlib.reload(definers)

        self.assertTrue(definers.has_sox())
        self.assertIs(definers.sox, cached_sox)
        mocked_import.assert_not_called()
