from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping, Sequence
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
    def get_os_name(self) -> str: ...

    def is_admin_windows(self) -> bool: ...

    def cores(self) -> int | None: ...

    def get_python_version(self) -> str | None: ...

    def importable(self, name: str) -> bool: ...

    def runnable(self, command: str) -> bool: ...

    def check_version_wildcard(
        self, version_spec: Any, version_actual: Any
    ) -> bool: ...

    def installed(self, package_name: str, version: str | None = None) -> bool: ...


class FileSystemPort(Protocol):
    def exist(self, *path_parts: str) -> bool: ...

    def load(self, path: str) -> Any: ...

    def read(self, path: str) -> Any: ...

    def save(self, path: str, text: Any = "") -> Any: ...

    def write(self, path: str, text: Any = "") -> Any: ...

    def directory(self, path: str, exist_ok: bool = True) -> Any: ...

    def copy(self, source: str, target: str) -> Any: ...

    def move(self, source: str, target: str) -> Any: ...

    def delete(self, path: PathInput) -> Any: ...

    def remove(self, path: PathInput) -> Any: ...

    def permit(
        self,
        path: str,
        *,
        exists_func: Any | None = None,
        get_os_name_func: Any | None = None,
        subprocess_module: Any | None = None,
    ) -> bool: ...


class ProcessPort(Protocol):
    def secure_command(self, command: CommandInput) -> list[str]: ...

    def run_linux(
        self,
        command: CommandInput,
        silent: bool = False,
        env: ProcessEnvironment = None,
    ) -> Any: ...

    def run_windows(
        self,
        command: CommandInput,
        silent: bool = False,
        env: ProcessEnvironment = None,
    ) -> Any: ...

    def run(
        self,
        command: CommandInput,
        silent: bool = False,
        env: ProcessEnvironment = None,
    ) -> Any: ...

    def get_process_pid(self, process_name: str) -> int | None: ...

    def send_signal_to_process(self, pid: int, signal_number: int) -> bool: ...


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
    def get_state(self, scope: RuntimeScope = "default") -> RuntimeStatePort: ...

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
