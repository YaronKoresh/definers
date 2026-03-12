import os
from pathlib import Path

import pytest

from definers._system import sanitize_load_path


def test_sanitize_path_basic(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    good = base / "model.pkl"
    good.write_text("ok")

    os.environ.pop("DEFINERS_TRUSTED_PATHS", None)
    with pytest.raises(ValueError):
        sanitize_load_path(str(tmp_path / "outside.pkl"))

    explicit = sanitize_load_path(str(good), allow_dirs=[str(base)])
    assert explicit == str(good.resolve())

    os.environ["DEFINERS_TRUSTED_PATHS"] = str(base)
    assert sanitize_load_path(str(good)) == str(good.resolve())
    with pytest.raises(ValueError):
        sanitize_load_path(str(tmp_path / "outside.pkl"))


def test_sanitize_path_tempdir_not_whitelisted(tmp_path):
    import tempfile

    tempdir = Path(tempfile.gettempdir()).resolve()
    outside = tempdir / "load_not_trusted.pkl"
    outside.write_text("x")
    os.environ["DEFINERS_TRUSTED_PATHS"] = ""
    with pytest.raises(ValueError):
        sanitize_load_path(str(outside))


def test_sanitize_path_allows_cwd(tmp_path, monkeypatch):

    file_in_cwd = tmp_path / "inside.txt"
    file_in_cwd.write_text("hello")

    monkeypatch.chdir(tmp_path)

    assert sanitize_load_path(str(file_in_cwd)) == str(file_in_cwd.resolve())


def test_sanitize_path_rejects_traversal(tmp_path):
    base = tmp_path / "a"
    base.mkdir()
    target = tmp_path / "b" / "m.pkl"
    target.parent.mkdir()
    target.write_text("x")
    os.environ["DEFINERS_TRUSTED_PATHS"] = str(base)

    with pytest.raises(ValueError):
        sanitize_load_path(str(base / "../b/m.pkl"))
