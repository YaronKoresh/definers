import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from definers.platform import compute


class TestPlatformComputeFree(unittest.TestCase):
    def test_preserves_default_huggingface_cache(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
        run_calls = []
        catch_calls = []

        with tempfile.TemporaryDirectory() as temp_root:
            cache_file = Path(temp_root) / ".cache" / "huggingface" / "marker.txt"
            cache_file.parent.mkdir(parents=True)
            cache_file.write_text("keep")

            with (
                patch.dict("sys.modules", {"torch": fake_torch}),
                patch.object(
                    compute.Path,
                    "home",
                    return_value=Path(temp_root),
                ),
            ):
                compute.free(
                    catch_func=catch_calls.append,
                    run_func=lambda *args, **kwargs: run_calls.append((args, kwargs)),
                    environ={},
                )

            self.assertTrue(cache_file.exists())
            self.assertEqual(catch_calls, [])

    def test_deletes_ephemeral_huggingface_cache(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
        run_calls = []
        catch_calls = []

        with tempfile.TemporaryDirectory() as temp_root:
            cache_dir = Path(temp_root) / "huggingface"
            cache_dir.mkdir()
            (cache_dir / "marker.txt").write_text("remove")

            with patch.dict("sys.modules", {"torch": fake_torch}):
                compute.free(
                    catch_func=catch_calls.append,
                    run_func=lambda *args, **kwargs: run_calls.append((args, kwargs)),
                    environ={"HF_HOME": str(cache_dir)},
                )

            self.assertFalse((cache_dir / "marker.txt").exists())
            self.assertEqual(catch_calls, [])