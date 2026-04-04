import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


def load_clean_workspace_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "clean_workspace.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "clean_workspace_module", module_path
    )
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)
    return module


clean_workspace = load_clean_workspace_module()


class TestCleanWorkspace(unittest.TestCase):
    def test_main_preserves_virtual_environment_binaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            protected_binary = (
                workspace_root
                / ".venv"
                / "Lib"
                / "site-packages"
                / "numpy"
                / "core"
                / "_multiarray_umath.pyd"
            )
            protected_binary.parent.mkdir(parents=True, exist_ok=True)
            protected_binary.write_bytes(b"binary")
            removable_cache = workspace_root / "src" / "pkg" / "__pycache__"
            removable_cache.mkdir(parents=True, exist_ok=True)
            (removable_cache / "module.pyc").write_bytes(b"cache")
            removable_build = workspace_root / "build"
            removable_build.mkdir(parents=True, exist_ok=True)
            (removable_build / "artifact.txt").write_text(
                "artifact", encoding="utf-8"
            )

            original_cwd = Path.cwd()
            os.chdir(workspace_root)
            try:
                clean_workspace.main()
            finally:
                os.chdir(original_cwd)

            self.assertTrue(protected_binary.exists())
            self.assertFalse(removable_cache.exists())
            self.assertFalse(removable_build.exists())

    def test_is_protected_path_matches_common_virtualenv_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir).resolve()

            for directory_name in (".venv", "venv", "env", ".env"):
                with self.subTest(directory_name=directory_name):
                    candidate = workspace_root / directory_name / "bin"
                    self.assertTrue(
                        clean_workspace.is_protected_path(
                            candidate, workspace_root
                        )
                    )


if __name__ == "__main__":
    unittest.main()
