import os

import pytest

from definers import git
from definers._system import sanitize_path


def test_sanitize_path_allows_and_rejects(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    good = base / "file.txt"
    good.write_text("x")
    os.environ["DEFINERS_TRUSTED_PATHS"] = str(base)
    assert sanitize_path(str(good)) == str(good.resolve())
    with pytest.raises(ValueError):
        sanitize_path(str(tmp_path / "other.txt"))


def test_sanitize_path_prevents_traversal(tmp_path):
    base = tmp_path / "a"
    base.mkdir()
    target = tmp_path / "b" / "foo"
    target.parent.mkdir()
    target.write_text("y")
    os.environ["DEFINERS_TRUSTED_PATHS"] = str(base)
    with pytest.raises(ValueError):
        sanitize_path(str(base / "../b/foo"))


def test_git_parent_untrusted(tmp_path):
    bad_parent = tmp_path / "bad"
    bad_parent.mkdir()
    os.environ["DEFINERS_TRUSTED_PATHS"] = str(tmp_path / "other")
    with pytest.raises(ValueError):
        git("user", "repo", parent=str(bad_parent))


def test_git_branch_and_run_list(monkeypatch, tmp_path):
    from definers import git, run

    calls = []

    def fake_run(arg, env=None):
        calls.append(arg)
        return []

    monkeypatch.setattr("definers._ml.run", fake_run)

    os.environ["DEFINERS_TRUSTED_PATHS"] = str(tmp_path)
    git("u", "r", branch="feature/x", parent=str(tmp_path))
    assert calls
    assert isinstance(calls[-1], list)
    assert calls[-1][0] == "git"
    assert "--branch" in calls[-1]

    with pytest.raises(ValueError):
        git("u", "r", branch="bad;rm -rf /", parent=str(tmp_path))


def test_find_latest_checkpoint_untrusted(tmp_path):
    from definers import find_latest_checkpoint

    base = tmp_path / "base"
    base.mkdir()
    os.environ["DEFINERS_TRUSTED_PATHS"] = str(tmp_path / "other")
    assert find_latest_checkpoint(str(base), "model") is None


def test_rvc_to_onnx_untrusted(tmp_path):
    from definers import rvc_to_onnx

    fake = tmp_path / "w.pth"
    fake.write_text("")
    os.environ["DEFINERS_TRUSTED_PATHS"] = str(tmp_path / "nothing")
    assert rvc_to_onnx(str(fake)) is None


def test_train_model_rvc_untrusted(tmp_path):
    from definers import train_model_rvc

    audio = tmp_path / "input.wav"
    audio.write_text("")
    os.environ["DEFINERS_TRUSTED_PATHS"] = str(tmp_path / "other")
    assert train_model_rvc("exp", str(audio)) is None


def test_convert_vocal_rvc_untrusted(tmp_path):
    from definers import convert_vocal_rvc

    audio = tmp_path / "input.wav"
    audio.write_text("")
    os.environ["DEFINERS_TRUSTED_PATHS"] = str(tmp_path / "nothing")
    assert convert_vocal_rvc("exp", str(audio)) is None


def test_convert_vocal_rvc_missing_deps(tmp_path):
    from definers import convert_vocal_rvc

    audio = tmp_path / "input.wav"
    audio.write_text("")

    assert convert_vocal_rvc("exp", str(audio)) is None


def test_sanitize_basename_and_experiment(tmp_path, capsys):
    from definers import convert_vocal_rvc, export_files_rvc, train_model_rvc
    from definers._system import sanitize_basename

    assert sanitize_basename("abc_123") == "abc_123"

    with pytest.raises(ValueError):
        sanitize_basename("../evil")
    with pytest.raises(ValueError):
        sanitize_basename("bad/name")
    with pytest.raises(ValueError):
        sanitize_basename("")

    assert export_files_rvc("good") == [] or isinstance(
        export_files_rvc("good"), list
    )
    assert export_files_rvc("../bad") == []
    assert train_model_rvc("../bad", str(tmp_path / "foo.wav")) is None
    assert convert_vocal_rvc("bad/name", str(tmp_path / "foo.wav")) is None
