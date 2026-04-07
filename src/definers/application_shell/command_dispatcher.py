from definers.application_shell.install_command import InstallCommand
from definers.application_shell.install_runtime_port import InstallRuntimePort
from definers.application_shell.lyric_video_command import LyricVideoCommand
from definers.application_shell.lyric_video_port import LyricVideoPort
from definers.application_shell.music_video_command import MusicVideoCommand
from definers.application_shell.music_video_port import MusicVideoPort
from definers.application_shell.output_port import OutputPort
from definers.application_shell.start_command import StartCommand
from definers.application_shell.start_project_port import StartProjectPort
from definers.application_shell.unknown_command import UnknownCommand


def _missing_install_handler(
    target: str,
    *,
    target_kind: str,
    list_only: bool,
    output: OutputPort,
) -> int:
    output("install command is not configured")
    return 1


class CliCommandDispatcher:
    @staticmethod
    def dispatch_cli_command(
        command: StartCommand
        | MusicVideoCommand
        | LyricVideoCommand
        | InstallCommand
        | UnknownCommand,
        *,
        start: StartProjectPort,
        music_video: MusicVideoPort,
        lyric_video: LyricVideoPort,
        install: InstallRuntimePort | None = None,
        output: OutputPort,
    ) -> int:
        install_handler = (
            _missing_install_handler if install is None else install
        )
        if isinstance(command, StartCommand):
            return int(start(command.project))
        if isinstance(command, MusicVideoCommand):
            output(
                music_video(
                    command.audio,
                    command.width,
                    command.height,
                    command.fps,
                )
            )
            return 0
        if isinstance(command, LyricVideoCommand):
            output(
                lyric_video(
                    command.audio,
                    command.background,
                    command.lyrics,
                    command.position,
                    max_dim=command.max_dim,
                    font_size=command.font_size,
                    text_color=command.text_color,
                    stroke_color=command.stroke_color,
                    stroke_width=command.stroke_width,
                    fade_duration=command.fade,
                )
            )
            return 0
        if isinstance(command, InstallCommand):
            return int(
                install_handler(
                    command.target,
                    target_kind=command.target_kind,
                    list_only=command.list_only,
                    output=output,
                )
            )
        output(f"unknown command {command.name}")
        return 1
