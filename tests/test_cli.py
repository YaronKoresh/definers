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
    gui_entrypoints = importlib.import_module("definers.ui.gui_entrypoints")

    assert gui_entrypoints.__name__ == "definers.ui.gui_entrypoints"
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


def test_install_list_outputs_known_targets():
    code, out = run_cli(["install", "--list"])

    assert code == 0
    assert "available install groups:" in out
    assert "available model domains:" in out
    assert "audio" in out
    assert "tts" in out
    assert "translate" in out
    assert "gradio" in out
    assert "stems" in out
    assert "fairseq" not in out


def test_install_group_dispatch(monkeypatch):
    import definers.cli.install as cli_install

    called = {}

    def fake_install(target, *, target_kind, list_only, output):
        called["target"] = target
        called["target_kind"] = target_kind
        called["list_only"] = list_only
        output("installed audio")
        return 0

    monkeypatch.setattr(
        cli_install,
        "run_optional_install_command",
        fake_install,
    )

    code, out = run_cli(["install", "audio"])

    assert code == 0
    assert out == "installed audio"
    assert called == {
        "target": "audio",
        "target_kind": "group",
        "list_only": False,
    }


def test_install_task_dispatch(monkeypatch):
    import definers.cli.install as cli_install

    called = {}

    def fake_install(target, *, target_kind, list_only, output):
        called["target"] = target
        called["target_kind"] = target_kind
        called["list_only"] = list_only
        return 0

    monkeypatch.setattr(
        cli_install,
        "run_optional_install_command",
        fake_install,
    )

    code, out = run_cli(["install", "translate", "--type", "task"])

    assert code == 0
    assert out == ""
    assert called == {
        "target": "translate",
        "target_kind": "task",
        "list_only": False,
    }


def test_install_model_domain_dispatch(monkeypatch):
    import definers.cli.install as cli_install

    called = {}

    def fake_install(target, *, target_kind, list_only, output):
        called["target"] = target
        called["target_kind"] = target_kind
        called["list_only"] = list_only
        output("installed model domain audio")
        return 0

    monkeypatch.setattr(
        cli_install,
        "run_optional_install_command",
        fake_install,
    )

    code, out = run_cli(["install", "audio", "--type", "model-domain"])

    assert code == 0
    assert out == "installed model domain audio"
    assert called == {
        "target": "audio",
        "target_kind": "model-domain",
        "list_only": False,
    }


def test_install_module_reports_pinned_madmom_source(monkeypatch):
    import definers.cli.install as cli_install

    monkeypatch.setattr(
        cli_install,
        "install_optional_target",
        lambda target, *, kind: target == "madmom" and kind == "module",
    )

    output_lines = []
    code = cli_install.run_optional_install_command(
        "madmom",
        target_kind="module",
        list_only=False,
        output=output_lines.append,
    )

    assert code == 0
    assert len(output_lines) == 1
    assert "installed module madmom:" in output_lines[0]
    assert "27f032e8947204902c675e5e341a3faf5dc86dae" in output_lines[0]


def test_install_module_reports_pinned_basic_pitch_source(monkeypatch):
    import definers.cli.install as cli_install

    monkeypatch.setattr(
        cli_install,
        "install_optional_target",
        lambda target, *, kind: target == "basic_pitch" and kind == "module",
    )

    output_lines = []
    code = cli_install.run_optional_install_command(
        "basic_pitch",
        target_kind="module",
        list_only=False,
        output=output_lines.append,
    )

    assert code == 0
    assert len(output_lines) == 1
    assert "installed module basic_pitch:" in output_lines[0]
    assert "830590229b32e30faebf1626f046bb9d0b80def7" in output_lines[0]


def test_start_dispatch(monkeypatch, tmp_path):

    import definers.ui.gui_entrypoints as gui_entrypoints

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
    import definers.ui.gui_entrypoints as gui_entrypoints

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
    import definers.ui.gui_entrypoints as gui_entrypoints

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


def test_install_without_target_requires_argument_or_list():
    code, out = run_cli(["install"])

    assert code == 1
    assert out == "install target is required unless --list is used"


def test_install_removed_fairseq_module_is_rejected():
    code, out = run_cli(["install", "fairseq", "--type", "module"])

    assert code == 1
    assert (
        out
        == "unknown module target fairseq; run 'definers install --list' to inspect available targets"
    )


def test_install_removed_unknown_model_task_is_rejected():
    code, out = run_cli(["install", "nope", "--type", "model-task"])

    assert code == 1
    assert (
        out
        == "unknown model-task target nope; run 'definers install --list' to inspect available targets"
    )


def test_start_dispatch_uses_registry_resolution(monkeypatch):
    import definers.ui.gui_entrypoints as gui_entrypoints

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
    import definers.ui.gui_entrypoints as gui_entrypoints

    monkeypatch.setitem(
        gui_entrypoints.__dict__,
        "_gui_namespace_only",
        lambda: 0,
    )

    code, out = run_cli(["namespace-only"])

    assert code == 1
    assert out == "unknown command namespace-only"
