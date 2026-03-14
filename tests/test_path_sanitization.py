import os
from pathlib import Path

import pytest

from definers.ml import (
    convert_vocal_rvc,
    export_files_rvc,
    find_latest_checkpoint,
    git,
    rvc_to_onnx,
    train_model_rvc,
)
from definers.system import secure_path


def test_sanitize_path_allows_and_rejects(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    good = base / "file.txt"
    good.write_text("x")

    assert secure_path(str(good), trust=str(base)) == str(good.resolve())

    other = tmp_path / "other.txt"
    other.write_text("x")
    assert secure_path(str(other)) == str(other.resolve())


def test_sanitize_path_tempdir_not_whitelisted(tmp_path):

    import tempfile

    tempdir = Path(tempfile.gettempdir()).resolve()

    outside = tempdir / "not_trusted.txt"

    outside.write_text("z")

    assert secure_path(str(outside)) == str(outside.resolve())


def test_sanitize_path_prevents_traversal(tmp_path):
    base = tmp_path / "a"
    base.mkdir()
    target = tmp_path / "b" / "foo"
    target.parent.mkdir()
    target.write_text("y")

    with pytest.raises(ValueError):
        secure_path(str(base / "../b/foo"), trust=str(base))


def test_git_branch_and_run_list(monkeypatch, tmp_path):
    from definers.system import run

    calls = []

    def fake_run(arg, env=None):
        calls.append(arg)
        return []

    monkeypatch.setattr("definers.ml.run", fake_run)

    os.environ["DEFINERS_TRUSTED_PATHS"] = str(tmp_path)
    git("u", "r", branch="feature/x", parent=str(tmp_path))
    assert calls
    assert isinstance(calls[-1], list)
    assert calls[-1][0] == "git"
    assert "--branch" in calls[-1]

    with pytest.raises(ValueError):
        git("u", "r", branch="bad;rm -rf /", parent=str(tmp_path))


def test_find_latest_checkpoint_untrusted(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    os.environ["DEFINERS_TRUSTED_PATHS"] = str(tmp_path / "other")
    assert find_latest_checkpoint(str(base), "model") is None


def test_rvc_to_onnx_untrusted(tmp_path):
    pytest.importorskip("definers.configs")
    fake = tmp_path / "w.pth"
    fake.write_text("")
    os.environ["DEFINERS_TRUSTED_PATHS"] = str(tmp_path / "nothing")
    assert rvc_to_onnx(str(fake)) is None


def test_train_model_rvc_untrusted(tmp_path):
    pytest.importorskip("definers.configs")
    audio = tmp_path / "input.wav"
    audio.write_text("")
    os.environ["DEFINERS_TRUSTED_PATHS"] = str(tmp_path / "other")
    assert train_model_rvc("exp", str(audio)) is None


def test_convert_vocal_rvc_untrusted(tmp_path):
    pytest.importorskip("definers.configs")
    audio = tmp_path / "input.wav"
    audio.write_text("")
    os.environ["DEFINERS_TRUSTED_PATHS"] = str(tmp_path / "nothing")
    assert convert_vocal_rvc("exp", str(audio)) is None


def test_convert_vocal_rvc_missing_deps(tmp_path):
    audio = tmp_path / "input.wav"
    audio.write_text("")

    assert convert_vocal_rvc("exp", str(audio)) is None


def test_sanitize_basename_and_experiment(tmp_path, capsys):
    assert secure_path("abc_123", basename=True) == "abc_123"

    with pytest.raises(ValueError):
        secure_path("../evil", basename=True)
    with pytest.raises(ValueError):
        secure_path("bad/name", basename=True)
    with pytest.raises(ValueError):
        secure_path("", basename=True)

    assert export_files_rvc("good") == [] or isinstance(
        export_files_rvc("good"), list
    )
    assert export_files_rvc("../bad") == []
    assert train_model_rvc("../bad", str(tmp_path / "foo.wav")) is None
    assert convert_vocal_rvc("bad/name", str(tmp_path / "foo.wav")) is None
