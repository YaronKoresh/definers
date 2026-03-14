from argparse import Namespace

from definers.application_shell.commands import (
    LyricVideoCommand,
    MusicVideoCommand,
    StartCommand,
    UnknownCommand,
    dispatch_cli_command,
    parse_cli_command,
)


def test_parse_cli_command_builds_start_command_from_default_start():
    command = parse_cli_command(
        Namespace(command=None, project="chat"),
        read_lyrics_text=lambda value: value,
        gui_commands=("chat", "video"),
    )

    assert command == StartCommand(project="chat")


def test_parse_cli_command_normalizes_gui_command_names():
    command = parse_cli_command(
        Namespace(command=" Video ", project=" Chat "),
        read_lyrics_text=lambda value: value,
        gui_commands=("chat", "video"),
    )

    assert command == StartCommand(project="video")


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
        gui_commands=("chat", "video"),
    )

    assert command == LyricVideoCommand(
        audio="a.mp3",
        background="b.png",
        lyrics="loaded:lyrics.txt",
        position="top",
        max_dim=640,
        font_size=70,
        text_color="white",
        stroke_color="black",
        stroke_width=2,
        fade=0.5,
    )


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


def test_parse_cli_command_normalizes_unknown_names():
    command = parse_cli_command(
        Namespace(command=" Nope ", project="chat"),
        read_lyrics_text=lambda value: value,
        gui_commands=("chat", "video"),
    )

    assert command == UnknownCommand(name="nope")