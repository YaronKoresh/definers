import os
import subprocess
import sys

import pytest

from definers import __version__


def run_cli(args, monkeypatch=None):

    from io import StringIO

    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    try:
        try:
            code = __import__("definers.cli").cli.main(args)
        except SystemExit as e:
            code = e.code
    finally:
        out = sys.stdout.getvalue().strip()
        err = sys.stderr.getvalue().strip()
        sys.stdout, sys.stderr = old_out, old_err

    combined = out if out else err
    return code, combined


def test_help():
    code, out = run_cli(["--help"])
    assert code == 0
    assert "usage:" in out.lower()


def test_version():
    code, out = run_cli(["--version"])
    assert code == 0
    assert __version__ in out


def test_start_dispatch(monkeypatch, tmp_path):

    import definers._chat as chat

    called = {}

    def fake_start(project=""):
        called["name"] = project
        return 0

    monkeypatch.setattr(chat, "start", fake_start)
    code, out = run_cli(["start", "video"])
    assert code == 0
    assert called["name"] == "video"

    code, out = run_cli(["audio"])
    assert code == 0
    assert called["name"] == "audio"


def test_music_video(monkeypatch):
    import definers._chat as chat

    monkeypatch.setattr(chat, "music_video", lambda a, w, h, f: "/tmp/x.mp4")
    code, out = run_cli(["music-video", "foo.mp3", "320", "240", "15"])
    assert code == 0
    assert out == "/tmp/x.mp4"


def test_lyric_video(monkeypatch, tmp_path, capsys):
    import definers._chat as chat

    monkeypatch.setattr(chat, "lyric_video", lambda *args, **k: "/tmp/y.mp4")

    lyrics_file = tmp_path / "lyrics.txt"
    lyrics_file.write_text("hello world")
    code, out = run_cli(
        ["lyric-video", "a.mp3", "b.png", str(lyrics_file), "top"]
    )
    assert code == 0
    assert out == "/tmp/y.mp4"


def test_unknown_command():
    code, out = run_cli(["nope"])
    assert code != 0
    assert "unknown command" in out.lower()
