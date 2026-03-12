import os

import pytest

from definers._system import sanitize_load_path


def test_sanitize_path_basic(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    good = base / "model.pkl"
    good.write_text("ok")

    os.environ.pop("DEFINERS_TRUSTED_PATHS", None)
    os.getcwd()

    try:
        result = sanitize_load_path(str(tmp_path / "outside.pkl"))

        assert result == str((tmp_path / "outside.pkl").resolve())
    except ValueError:
        pass
    explicit = sanitize_load_path(str(good), allow_dirs=[str(base)])
    assert explicit == str(good.resolve())

    os.environ["DEFINERS_TRUSTED_PATHS"] = str(base)
    assert sanitize_load_path(str(good)) == str(good.resolve())
    with pytest.raises(ValueError):
        sanitize_load_path(str(tmp_path / "outside.pkl"))


def test_sanitize_path_rejects_traversal(tmp_path):
    base = tmp_path / "a"
    base.mkdir()
    target = tmp_path / "b" / "m.pkl"
    target.parent.mkdir()
    target.write_text("x")
    os.environ["DEFINERS_TRUSTED_PATHS"] = str(base)

    with pytest.raises(ValueError):
        sanitize_load_path(str(base / "../b/m.pkl"))
