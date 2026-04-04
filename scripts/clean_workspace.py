import shutil
from pathlib import Path

PROTECTED_DIRECTORY_NAMES = (
    ".git",
    ".venv",
    "venv",
    "env",
    ".env",
)

DIRECTORY_PATTERNS = (
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
)

FILE_PATTERNS = (
    ".coverage",
    ".coverage.*",
    ".sqlite",
    "nosetests.xml",
    "coverage.xml",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".DS_Store",
)


def matches_any(path: Path, patterns: tuple[str, ...]) -> bool:
    return path.name in patterns or any(
        path.match(pattern) for pattern in patterns
    )


def is_protected_path(path: Path, workspace_root: Path) -> bool:
    relative_parts = path.relative_to(workspace_root).parts
    return any(part in PROTECTED_DIRECTORY_NAMES for part in relative_parts)


def remove_path(path: Path) -> None:
    if path.is_symlink():
        path.unlink(missing_ok=True)
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        return
    if path.is_file():
        path.unlink(missing_ok=True)


def main() -> None:
    workspace_root = Path(".").resolve()
    for path in workspace_root.rglob("*"):
        if is_protected_path(path, workspace_root):
            continue
        if matches_any(path, DIRECTORY_PATTERNS):
            remove_path(path)
            continue
        if matches_any(path, FILE_PATTERNS):
            remove_path(path)


if __name__ == "__main__":
    main()
