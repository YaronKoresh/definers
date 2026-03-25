class ApplicationShellFacade:
    @staticmethod
    def get_commands_module():
        import importlib

        return importlib.import_module("definers.application_shell.commands")

    @classmethod
    def get_command_export(cls, name):
        return getattr(cls.get_commands_module(), name)


CliCommand = ApplicationShellFacade.get_command_export("CliCommand")
CliRequest = ApplicationShellFacade.get_command_export("CliRequest")
LyricVideoCommand = ApplicationShellFacade.get_command_export(
    "LyricVideoCommand"
)
MusicVideoCommand = ApplicationShellFacade.get_command_export(
    "MusicVideoCommand"
)
StartCommand = ApplicationShellFacade.get_command_export("StartCommand")
UnknownCommand = ApplicationShellFacade.get_command_export("UnknownCommand")
dispatch_cli_command = ApplicationShellFacade.get_command_export(
    "dispatch_cli_command"
)
parse_cli_command = ApplicationShellFacade.get_command_export(
    "parse_cli_command"
)

__all__ = [
    "CliCommand",
    "CliRequest",
    "LyricVideoCommand",
    "MusicVideoCommand",
    "StartCommand",
    "UnknownCommand",
    "dispatch_cli_command",
    "parse_cli_command",
]
