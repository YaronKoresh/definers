from __future__ import annotations

import shutil
from pathlib import Path

DIRECTORY_PATTERNS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".tox",
    ".nox",
    ".cache",
    ".ipynb_checkpoints",
    "htmlcov",
    "build",
    "dist",
    ".eggs",
    "*.egg-info",
    ".coverage.reports",
}

FILE_PATTERNS = {
    ".coverage",
    ".coverage.*",
    ".sqlite",
    "nosetests.xml",
    "coverage.xml",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".DS_Store",
}


def should_match(patterns: set[str], candidate: Path) -> bool:
    for pattern in patterns:
        if candidate.match(pattern):
            return True
    return False


def remove_artifacts(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_dir() and should_match(DIRECTORY_PATTERNS, path):
            shutil.rmtree(path, ignore_errors=True)
            continue
        if path.is_file() and should_match(FILE_PATTERNS, path):
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    remove_artifacts(Path("."))
