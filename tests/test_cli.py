import importlib
import sys

from definers import __version__
from definers.cli import main as cli_main


def run_cli(args, monkeypatch=None):

    from io import StringIO

    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    try:
        try:
            code = cli_main(args)
        except SystemExit as e:
            code = e.code
    finally:
        out = sys.stdout.getvalue().strip()
        err = sys.stderr.getvalue().strip()
        sys.stdout, sys.stderr = old_out, old_err

    combined = out if out else err
    return code, combined


def test_package_version_matches_top_level_version():
    definers_package = importlib.import_module("definers")

    assert hasattr(definers_package, "__version__")
    assert definers_package.__version__ == __version__


def test_chat_module_import_surface():
    gui_entrypoints = importlib.import_module(
        "definers.presentation.gui_entrypoints"
    )

    assert (
        gui_entrypoints.__name__ == "definers.presentation.gui_entrypoints"
    )
    assert hasattr(gui_entrypoints, "start")
    assert hasattr(gui_entrypoints, "_gui_chat")


def test_help():
    definers_package = importlib.import_module("definers")

    assert definers_package.__version__ == __version__

    code, out = run_cli(["--help"])
    assert code == 0
    assert "usage:" in out.lower()


def test_version():
    definers_package = importlib.import_module("definers")

    assert definers_package.__version__ == __version__

    code, out = run_cli(["--version"])
    assert code == 0
    assert __version__ in out


def test_start_dispatch(monkeypatch, tmp_path):

    import definers.presentation.gui_entrypoints as gui_entrypoints

    assert hasattr(gui_entrypoints, "start")

    called = {}

    def fake_start(project=""):
        called["name"] = project
        return 0

    monkeypatch.setattr(gui_entrypoints, "start", fake_start)
    code, out = run_cli(["start", "video"])
    assert code == 0
    assert called["name"] == "video"

    code, out = run_cli(["audio"])
    assert code == 0
    assert called["name"] == "audio"


def test_music_video(monkeypatch):
    import definers.presentation.gui_entrypoints as gui_entrypoints

    assert hasattr(gui_entrypoints, "music_video")

    monkeypatch.setattr(
        gui_entrypoints,
        "music_video",
        lambda a, w, h, f: "/tmp/x.mp4",
    )
    code, out = run_cli(["music-video", "foo.mp3", "320", "240", "15"])
    assert code == 0
    assert out == "/tmp/x.mp4"


def test_lyric_video(monkeypatch, tmp_path, capsys):
    import definers.presentation.gui_entrypoints as gui_entrypoints

    assert hasattr(gui_entrypoints, "lyric_video")

    monkeypatch.setattr(
        gui_entrypoints,
        "lyric_video",
        lambda *args, **k: "/tmp/y.mp4",
    )

    lyrics_file = tmp_path / "lyrics.txt"
    lyrics_file.write_text("hello world")
    code, out = run_cli(
        ["lyric-video", "a.mp3", "b.png", str(lyrics_file), "top"]
    )
    assert code == 0
    assert out == "/tmp/y.mp4"


def test_unknown_command():
    definers_package = importlib.import_module("definers")

    assert definers_package.__version__ == __version__

    code, out = run_cli(["nope"])
    assert code != 0
    assert "unknown command" in out.lower()


def test_start_dispatch_uses_registry_resolution(monkeypatch):
    import definers.presentation.gui_entrypoints as gui_entrypoints

    assert hasattr(gui_entrypoints, "_gui_chat")

    called = []

    monkeypatch.setattr(
        gui_entrypoints,
        "_gui_chat",
        lambda: called.append("chat") or 0,
    )

    code, out = run_cli(["chat"])

    assert code == 0
    assert called == ["chat"]


def test_unknown_namespace_only_gui_command_stays_invalid(monkeypatch):
    import definers.presentation.gui_entrypoints as gui_entrypoints

    monkeypatch.setitem(
        gui_entrypoints.__dict__,
        "_gui_namespace_only",
        lambda: 0,
    )

    code, out = run_cli(["namespace-only"])

    assert code == 1
    assert out == "unknown command namespace-only"
