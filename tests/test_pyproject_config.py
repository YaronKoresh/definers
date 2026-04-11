from pathlib import Path


def read_pyproject_text() -> str:
    return (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )


def read_doc_text(name: str) -> str:
    return (Path(__file__).resolve().parents[1] / name).read_text(
        encoding="utf-8"
    )


def test_pyproject_omits_fairseq_and_vcs_install_metadata():
    pyproject_text = read_pyproject_text()

    assert "fairseq @" not in pyproject_text
    assert "fairseq2" not in pyproject_text
    assert "git+https://" not in pyproject_text
    assert "faiss @ https://" not in pyproject_text
    assert "allow-direct-references" not in pyproject_text


def test_pyproject_declares_clean_coverage_task():
    pyproject_text = read_pyproject_text()

    assert (
        'coverage = ["_coverage-clean", "_coverage-run", "_coverage-clean"]'
        in pyproject_text
    )
    assert "python -m pytest -q" in pyproject_text
    assert "--cov=src/definers" in pyproject_text
    assert "--cov-branch" in pyproject_text
    assert "--cov-report=term-missing:skip-covered" in pyproject_text
    assert 'COVERAGE_FILE = ".coverage.poe"' in pyproject_text
    assert "Path('.coverage.poe').unlink(missing_ok=True)" in pyproject_text


def test_pyproject_declares_python_314_support():
    pyproject_text = read_pyproject_text()

    assert 'requires-python = ">=3.10,<3.15"' in pyproject_text
    assert '"Programming Language :: Python :: 3.13"' in pyproject_text
    assert '"Programming Language :: Python :: 3.14"' in pyproject_text
