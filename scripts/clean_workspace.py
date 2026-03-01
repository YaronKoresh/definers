import pathlib
import shutil

targets = {
    "directories": {
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
    },
    "files": {
        ".coverage",
        ".coverage.*",
        ".sqlite",
        "nosetests.xml",
        "coverage.xml",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".DS_Store",
    },
}
for p in pathlib.Path(".").rglob("*"):
    if p.name in targets["directories"] or any(
        p.match(pat) for pat in targets["directories"]
    ):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    elif p.name in targets["files"] or any(
        p.match(pat) for pat in targets["files"]
    ):
        if p.is_file():
            p.unlink(missing_ok=True)
