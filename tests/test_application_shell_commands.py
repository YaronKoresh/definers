from argparse import Namespace

from definers.cli.command_dispatcher import dispatch_cli_command
from definers.cli.command_parser import parse_cli_command
from definers.cli.command_registry import (
    create_cli_command_registry,
    get_known_cli_names,
)
from definers.cli.install_command import InstallCommand
from definers.cli.lyric_video_command import LyricVideoCommand
from definers.cli.music_video_command import MusicVideoCommand
from definers.cli.request_coercer import coerce_cli_request
from definers.cli.start_command import StartCommand
from definers.cli.unknown_command import UnknownCommand

COMMAND_REGISTRY = create_cli_command_registry(("chat", "video"))


def test_parse_cli_command_builds_start_command_from_default_start():
    command = parse_cli_command(
        Namespace(command=None, project="chat"),
        read_lyrics_text=lambda value: value,
        command_registry=COMMAND_REGISTRY,
    )

    assert isinstance(command, StartCommand)
    assert command.project == "chat"
    assert command.metadata is not None
    assert command.metadata.resolved_name == "start"


def test_parse_cli_command_normalizes_gui_command_names():
    command = parse_cli_command(
        Namespace(command=" Video ", project=" Chat "),
        read_lyrics_text=lambda value: value,
        command_registry=COMMAND_REGISTRY,
    )

    assert isinstance(command, StartCommand)
    assert command.project == "video"
    assert command.metadata is not None
    assert command.metadata.resolved_name == "video"


def test_parse_cli_command_builds_lyric_video_command_with_loaded_lyrics():
    command = parse_cli_command(
        Namespace(
            command="lyric-video",
            audio="a.mp3",
            background="b.png",
            lyrics="lyrics.txt",
            position="top",
            max_dim=640,
            font_size=70,
            text_color="white",
            stroke_color="black",
            stroke_width=2,
            fade=0.5,
        ),
        read_lyrics_text=lambda value: f"loaded:{value}",
        command_registry=COMMAND_REGISTRY,
    )

    assert isinstance(command, LyricVideoCommand)
    assert command.audio == "a.mp3"
    assert command.background == "b.png"
    assert command.lyrics == "loaded:lyrics.txt"
    assert command.position == "top"
    assert command.max_dim == 640
    assert command.font_size == 70
    assert command.text_color == "white"
    assert command.stroke_color == "black"
    assert command.stroke_width == 2
    assert command.fade == 0.5
    assert command.metadata is not None
    assert command.metadata.resolved_name == "lyric-video"


def test_dispatch_cli_command_writes_music_video_result():
    outputs: list[object] = []

    exit_code = dispatch_cli_command(
        MusicVideoCommand(audio="a.mp3", width=320, height=240, fps=15),
        start=lambda project: 0,
        music_video=lambda audio, width, height, fps: (
            audio,
            width,
            height,
            fps,
        ),
        lyric_video=lambda *args, **kwargs: None,
        output=outputs.append,
    )

    assert exit_code == 0
    assert outputs == [("a.mp3", 320, 240, 15)]


def test_dispatch_cli_command_handles_unknown_commands():
    outputs: list[object] = []

    exit_code = dispatch_cli_command(
        UnknownCommand(name="nope"),
        start=lambda project: 0,
        music_video=lambda audio, width, height, fps: None,
        lyric_video=lambda *args, **kwargs: None,
        output=outputs.append,
    )

    assert exit_code == 1
    assert outputs == ["unknown command nope"]


def test_dispatch_cli_command_rejects_install_without_handler():
    outputs: list[object] = []

    exit_code = dispatch_cli_command(
        InstallCommand(target="audio", target_kind="group", list_only=False),
        start=lambda project: 0,
        music_video=lambda audio, width, height, fps: None,
        lyric_video=lambda *args, **kwargs: None,
        output=outputs.append,
    )

    assert exit_code == 1
    assert outputs == ["install command is not configured"]


def test_parse_cli_command_normalizes_unknown_names():
    command = parse_cli_command(
        Namespace(command=" Nope ", project="chat"),
        read_lyrics_text=lambda value: value,
        command_registry=COMMAND_REGISTRY,
    )

    assert isinstance(command, UnknownCommand)
    assert command.name == "nope"
    assert command.metadata is not None
    assert command.metadata.resolved_name is None


def test_create_cli_command_registry_rejects_builtin_command_conflicts():
    try:
        create_cli_command_registry(("music-video",))
    except ValueError as exc:
        assert str(exc) == "CLI command name conflict: music-video"
    else:
        raise AssertionError("expected ValueError")


def test_get_known_cli_names_can_exclude_option_flags():
    assert get_known_cli_names(
        COMMAND_REGISTRY,
        include_options=False,
    ) == (
        "start",
        "music-video",
        "lyric-video",
        "chat",
        "video",
    )


def test_coerce_cli_request_uses_namespace_defaults():
    request = coerce_cli_request(Namespace(command="start"))

    assert request.project == "chat"
    assert request.position == "bottom"
    assert request.fade == 0.5
