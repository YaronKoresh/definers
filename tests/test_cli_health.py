from definers.cli.application.catalog import create_cli_command_registry
from definers.cli.application.runtime import resolve_cli_handlers
from definers.cli.health import (
    collect_cli_health_snapshot,
    run_cli_health_check,
    validate_cli_health_snapshot,
)


def test_collect_cli_health_snapshot_reports_registry_metrics():
    command_registry = create_cli_command_registry(["chat", "audio"])

    snapshot = collect_cli_health_snapshot(
        command_registry=command_registry,
        gui_project_names=("chat", "audio"),
    )

    assert snapshot.command_names == (
        "audio",
        "chat",
        "install",
        "lyric-video",
        "music-video",
        "start",
    )
    assert snapshot.gui_project_names == ("audio", "chat")
    assert snapshot.direct_gui_command_names == ("audio", "chat")
    assert snapshot.media_command_names == (
        "install",
        "lyric-video",
        "music-video",
    )
    assert snapshot.command_count == 6
    assert snapshot.gui_project_count == 2
    assert snapshot.known_names_with_options[-2:] == ("--help", "--version")


def test_validate_cli_health_snapshot_rejects_missing_direct_commands():
    command_registry = create_cli_command_registry(["chat"])
    snapshot = collect_cli_health_snapshot(
        command_registry=command_registry,
        gui_project_names=("chat", "audio"),
    )

    try:
        validate_cli_health_snapshot(snapshot)
    except LookupError as exc:
        assert str(exc) == "Missing direct GUI commands: audio"
    else:
        raise AssertionError("expected LookupError")


def test_run_cli_health_check_matches_live_registry():
    snapshot = run_cli_health_check()

    assert "start" in snapshot.command_names
    assert "install" in snapshot.command_names
    assert "music-video" in snapshot.command_names
    assert "lyric-video" in snapshot.command_names
    assert "chat" in snapshot.gui_project_names
    assert set(snapshot.gui_project_names).issubset(
        snapshot.direct_gui_command_names
    )


def test_resolve_cli_handlers_returns_callables():
    start, music_video, lyric_video = resolve_cli_handlers()

    assert callable(start)
    assert callable(music_video)
    assert callable(lyric_video)
