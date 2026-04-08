from typing import Protocol

from definers.application_shell.output_port import OutputPort


class InstallRuntimePort(Protocol):
    def __call__(
        self,
        target: str,
        *,
        target_kind: str,
        list_only: bool,
        output: OutputPort,
    ) -> int: ...
