import shutil
from pathlib import Path

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
    for path in Path(".").rglob("*"):
        if matches_any(path, DIRECTORY_PATTERNS):
            remove_path(path)
            continue
        if matches_any(path, FILE_PATTERNS):
            remove_path(path)


if __name__ == "__main__":
    main()
