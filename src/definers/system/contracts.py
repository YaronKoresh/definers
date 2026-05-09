from __future__ import annotations

from collections.abc import (
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)
from typing import Any, Protocol, TypeAlias

CommandArguments: TypeAlias = Sequence[str]
CommandInput: TypeAlias = str | CommandArguments
PathParts: TypeAlias = Sequence[str]
PathInput: TypeAlias = str | PathParts
ProcessEnvironment: TypeAlias = Mapping[str, str] | None
RuntimeScope: TypeAlias = str
RuntimeValueMap: TypeAlias = MutableMapping[str, Any]
TokenizerEntryMap: TypeAlias = MutableMapping[str, Any]
TokenizerMap: TypeAlias = MutableMapping[str, TokenizerEntryMap]


class EnvironmentPort(Protocol):
    get_os_name: Callable[[], str]
    is_admin_windows: Callable[[], bool]
    cores: Callable[[], int | None]
    get_python_version: Callable[[], str | None]
    importable: Callable[[str], bool]
    runnable: Callable[[str], bool]
    check_version_wildcard: Callable[[Any, Any], bool]
    installed: Callable[[str, str | None], bool]


class FileSystemPort(Protocol):
    exist: Callable[..., bool]
    load: Callable[[str], Any]
    read: Callable[[str], Any]
    save: Callable[[str, Any], Any]
    write: Callable[[str, Any], Any]
    directory: Callable[[str, bool], Any]
    copy: Callable[[str, str], Any]
    move: Callable[[str, str], Any]
    delete: Callable[[PathInput], Any]
    remove: Callable[[PathInput], Any]
    permit: Callable[..., bool]


class ProcessPort(Protocol):
    secure_command: Callable[[CommandInput], list[str]]
    run_linux: Callable[[CommandInput, bool, ProcessEnvironment], Any]
    run_windows: Callable[[CommandInput, bool, ProcessEnvironment], Any]
    run: Callable[[CommandInput, bool, ProcessEnvironment], Any]
    get_process_pid: Callable[[str], int | None]
    send_signal_to_process: Callable[[int, int], bool]


class InfrastructurePorts(Protocol):
    environment: EnvironmentPort
    filesystem: FileSystemPort
    processes: ProcessPort


class InfrastructureServiceContextPort(Protocol):
    def get(self) -> InfrastructurePorts: ...

    def set(self, services: InfrastructurePorts) -> InfrastructurePorts: ...

    def reset(self) -> InfrastructurePorts: ...


class RuntimeCollectionsPort(Protocol):
    models: RuntimeValueMap
    tokenizers: TokenizerMap
    processors: RuntimeValueMap
    configs: RuntimeValueMap


class RuntimeStatePort(RuntimeCollectionsPort, Protocol):
    def get_model(self, name: str, default: Any = None) -> Any: ...

    def set_model(self, name: str, value: Any) -> Any: ...

    def get_tokenizer_entry(
        self, name: str, default: dict[str, Any] | None = None
    ) -> dict[str, Any] | None: ...

    def get_tokenizer(self, name: str, default: Any = None) -> Any: ...

    def set_tokenizer(
        self,
        name: str,
        tokenizer: Any,
        *,
        model_name: str | None = None,
    ) -> dict[str, Any]: ...

    def get_processor(self, name: str, default: Any = None) -> Any: ...

    def set_processor(self, name: str, value: Any) -> Any: ...

    def get_config(self, name: str, default: Any = None) -> Any: ...

    def set_config(self, name: str, value: Any) -> Any: ...

    def reset(self) -> None: ...


class RuntimeStateRegistryPort(Protocol):
    def get_state(
        self, scope: RuntimeScope = "default"
    ) -> RuntimeStatePort: ...

    def create_state(
        self,
        scope: RuntimeScope,
        *,
        replace: bool = False,
    ) -> RuntimeStatePort: ...

    def delete_state(self, scope: RuntimeScope) -> None: ...

    def list_scopes(self) -> tuple[str, ...]: ...

    def reset(self, scope: RuntimeScope = "default") -> RuntimeStatePort: ...

    def reset_many(self, scopes: Iterable[str]) -> None: ...
