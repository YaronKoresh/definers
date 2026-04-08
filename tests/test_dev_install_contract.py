from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_TOP_LEVEL_TEST_IMPORTS = {
    "cv2",
    "datasets",
    "diffusers",
    "fastapi",
    "faiss",
    "gradio",
    "librosa",
    "matplotlib",
    "moviepy",
    "pydub",
    "sklearn",
    "soundfile",
    "tensorflow",
    "torch",
    "torchaudio",
    "transformers",
}


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _top_level_imports(file_path: Path) -> set[str]:
    imported_modules: set[str] = set()
    module_ast = ast.parse(file_path.read_text(encoding="utf-8"))
    for node in module_ast.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.split(".", 1)[0])
    return imported_modules


def test_ci_workflows_install_only_dev_extra():
    workflow_root = _workspace_root() / ".github" / "workflows"
    for workflow_name in ("check.yml", "quality.yml"):
        workflow_text = (workflow_root / workflow_name).read_text(
            encoding="utf-8"
        )
        assert 'pip install -e ".[dev]"' in workflow_text
        assert 'pip install -e ".[dev,all]"' not in workflow_text


def test_tests_avoid_top_level_optional_dependency_imports():
    tests_root = _workspace_root() / "tests"
    for test_file in tests_root.glob("test_*.py"):
        imported_modules = _top_level_imports(test_file)
        forbidden_imports = sorted(
            imported_modules & FORBIDDEN_TOP_LEVEL_TEST_IMPORTS
        )
        assert forbidden_imports == [], (
            f"{test_file.name} imports optional packages at module scope: "
            f"{', '.join(forbidden_imports)}"
        )
