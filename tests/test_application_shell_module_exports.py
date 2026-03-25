from argparse import Namespace

from definers.application_shell import (
    CliRequest,
    LyricVideoCommand,
    MusicVideoCommand,
    StartCommand,
    UnknownCommand,
    dispatch_cli_command,
    parse_cli_command,
)
from definers.application_shell.commands import create_cli_command_registry


def test_application_shell_facade_exports_parser_and_dispatcher():
    command_registry = create_cli_command_registry(("chat", "video"))
    command = parse_cli_command(
        Namespace(command=" video ", project="chat"),
        read_lyrics_text=lambda value: value,
        command_registry=command_registry,
    )

    assert isinstance(command, StartCommand)
    assert command.project == "video"

    outputs: list[object] = []
    exit_code = dispatch_cli_command(
        MusicVideoCommand(audio="a.mp3", width=320, height=240, fps=15),
        start=lambda project: 0,
        music_video=lambda audio, width, height, fps: (audio, width, height, fps),
        lyric_video=lambda *args, **kwargs: None,
        output=outputs.append,
    )

    assert exit_code == 0
    assert outputs == [("a.mp3", 320, 240, 15)]


def test_application_shell_facade_exports_command_types():
    request = CliRequest(command="start")
    lyric_command = LyricVideoCommand(
        audio="audio.mp3",
        background="background.png",
        lyrics="lyrics",
        position="bottom",
        max_dim=640,
        font_size=70,
        text_color="white",
        stroke_color="black",
        stroke_width=2,
        fade=0.5,
    )
    unknown = UnknownCommand(name="missing")

    assert request.command == "start"
    assert lyric_command.background == "background.png"
    assert unknown.name == "missing"