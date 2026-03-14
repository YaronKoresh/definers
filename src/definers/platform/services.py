from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from definers.platform import filesystem as _filesystem
from definers.platform import processes as _processes
from definers.platform import runtime as _runtime
from definers.platform.contracts import (
    CommandInput,
    EnvironmentPort,
    FileSystemPort,
    InfrastructureServiceContextPort,
    InfrastructurePorts,
    PathInput,
    ProcessEnvironment,
    ProcessPort,
)


@dataclass(slots=True)
class EnvironmentService:
    get_os_name_fn: Callable[[], str] = _runtime.get_os_name
    is_admin_windows_fn: Callable[[], bool] = _runtime.is_admin_windows
    cores_fn: Callable[[], int | None] = _runtime.cores
    get_python_version_fn: Callable[[], str | None] = _runtime.get_python_version
    importable_fn: Callable[[str], bool] = _runtime.importable
    runnable_fn: Callable[[str], bool] = _runtime.runnable
    check_version_wildcard_fn: Callable[[Any, Any], bool] = _runtime.check_version_wildcard
    installed_fn: Callable[[str, str | None], bool] = _runtime.installed

    def get_os_name(self) -> str:
        return self.get_os_name_fn()

    def is_admin_windows(self) -> bool:
        return self.is_admin_windows_fn()

    def cores(self) -> int | None:
        return self.cores_fn()

    def get_python_version(self) -> str | None:
        return self.get_python_version_fn()

    def importable(self, name: str) -> bool:
        return self.importable_fn(name)

    def runnable(self, command: str) -> bool:
        return self.runnable_fn(command)

    def check_version_wildcard(self, version_spec: Any, version_actual: Any) -> bool:
        return self.check_version_wildcard_fn(version_spec, version_actual)

    def installed(self, package_name: str, version: str | None = None) -> bool:
        return self.installed_fn(package_name, version)


@dataclass(slots=True)
class FileSystemService:
    exist_fn: Callable[..., bool] = _filesystem.exist
    load_fn: Callable[[str], Any] = _filesystem.load
    read_fn: Callable[[str], Any] = _filesystem.read
    save_fn: Callable[[str, Any], Any] = _filesystem.save
    write_fn: Callable[[str, Any], Any] = _filesystem.write
    directory_fn: Callable[[str, bool], Any] = _filesystem.directory
    copy_fn: Callable[[str, str], Any] = _filesystem.copy
    move_fn: Callable[[str, str], Any] = _filesystem.move
    delete_fn: Callable[[PathInput], Any] = _filesystem.delete
    remove_fn: Callable[[PathInput], Any] = _filesystem.remove
    permit_fn: Callable[..., bool] = _filesystem.permit

    def exist(self, *path_parts: str) -> bool:
        return self.exist_fn(*path_parts)

    def load(self, path: str) -> Any:
        return self.load_fn(path)

    def read(self, path: str) -> Any:
        return self.read_fn(path)

    def save(self, path: str, text: Any = "") -> Any:
        return self.save_fn(path, text)

    def write(self, path: str, text: Any = "") -> Any:
        return self.write_fn(path, text)

    def directory(self, path: str, exist_ok: bool = True) -> Any:
        return self.directory_fn(path, exist_ok)

    def copy(self, source: str, target: str) -> Any:
        return self.copy_fn(source, target)

    def move(self, source: str, target: str) -> Any:
        return self.move_fn(source, target)

    def delete(self, path: PathInput) -> Any:
        return self.delete_fn(path)

    def remove(self, path: PathInput) -> Any:
        return self.remove_fn(path)

    def permit(
        self,
        path: str,
        *,
        exists_func: Callable[..., bool] | None = None,
        get_os_name_func: Callable[[], str] | None = None,
        subprocess_module: Any | None = None,
    ) -> bool:
        kwargs: dict[str, Any] = {}
        if exists_func is not None:
            kwargs["exists_func"] = exists_func
        if get_os_name_func is not None:
            kwargs["get_os_name_func"] = get_os_name_func
        if subprocess_module is not None:
            kwargs["subprocess_module"] = subprocess_module
        return self.permit_fn(path, **kwargs)


@dataclass(slots=True)
class ProcessService:
    secure_command_fn: Callable[[CommandInput], list[str]] = _processes.secure_command
    run_linux_fn: Callable[[CommandInput, bool, ProcessEnvironment], Any] = _processes.run_linux
    run_windows_fn: Callable[[CommandInput, bool, ProcessEnvironment], Any] = _processes.run_windows
    run_fn: Callable[[CommandInput, bool, ProcessEnvironment], Any] = _processes.run
    get_process_pid_fn: Callable[[str], int | None] = _processes.get_process_pid
    send_signal_to_process_fn: Callable[[int, int], bool] = _processes.send_signal_to_process

    def secure_command(self, command: CommandInput) -> list[str]:
        return self.secure_command_fn(command)

    def run_linux(
        self,
        command: CommandInput,
        silent: bool = False,
        env: ProcessEnvironment = None,
    ) -> Any:
        return self.run_linux_fn(command, silent, env)

    def run_windows(
        self,
        command: CommandInput,
        silent: bool = False,
        env: ProcessEnvironment = None,
    ) -> Any:
        return self.run_windows_fn(command, silent, env)

    def run(
        self,
        command: CommandInput,
        silent: bool = False,
        env: ProcessEnvironment = None,
    ) -> Any:
        return self.run_fn(command, silent, env)

    def get_process_pid(self, process_name: str) -> int | None:
        return self.get_process_pid_fn(process_name)

    def send_signal_to_process(self, pid: int, signal_number: int) -> bool:
        return self.send_signal_to_process_fn(pid, signal_number)


@dataclass(slots=True)
class InfrastructureServices(InfrastructurePorts):
    environment: EnvironmentService = field(default_factory=EnvironmentService)
    filesystem: FileSystemService = field(default_factory=FileSystemService)
    processes: ProcessService = field(default_factory=ProcessService)


def _coerce_environment_service(
    environment: EnvironmentPort | None,
) -> EnvironmentService:
    if environment is None:
        return EnvironmentService()
    if isinstance(environment, EnvironmentService):
        return environment
    return EnvironmentService(
        get_os_name_fn=environment.get_os_name,
        is_admin_windows_fn=environment.is_admin_windows,
        cores_fn=environment.cores,
        get_python_version_fn=environment.get_python_version,
        importable_fn=environment.importable,
        runnable_fn=environment.runnable,
        check_version_wildcard_fn=environment.check_version_wildcard,
        installed_fn=environment.installed,
    )


def _coerce_filesystem_service(
    filesystem: FileSystemPort | None,
) -> FileSystemService:
    if filesystem is None:
        return FileSystemService()
    if isinstance(filesystem, FileSystemService):
        return filesystem
    return FileSystemService(
        exist_fn=filesystem.exist,
        load_fn=filesystem.load,
        read_fn=filesystem.read,
        save_fn=filesystem.save,
        write_fn=filesystem.write,
        directory_fn=filesystem.directory,
        copy_fn=filesystem.copy,
        move_fn=filesystem.move,
        delete_fn=filesystem.delete,
        remove_fn=filesystem.remove,
        permit_fn=filesystem.permit,
    )


def _coerce_process_service(processes: ProcessPort | None) -> ProcessService:
    if processes is None:
        return ProcessService()
    if isinstance(processes, ProcessService):
        return processes
    return ProcessService(
        secure_command_fn=processes.secure_command,
        run_linux_fn=processes.run_linux,
        run_windows_fn=processes.run_windows,
        run_fn=processes.run,
        get_process_pid_fn=processes.get_process_pid,
        send_signal_to_process_fn=processes.send_signal_to_process,
    )


def build_infrastructure_services(
    *,
    environment: EnvironmentPort | None = None,
    filesystem: FileSystemPort | None = None,
    processes: ProcessPort | None = None,
) -> InfrastructureServices:
    env = _coerce_environment_service(environment)
    fs = _coerce_filesystem_service(filesystem)
    proc = _coerce_process_service(processes)
    return InfrastructureServices(
        environment=env,
        filesystem=fs,
        processes=proc,
    )


def _coerce_infrastructure_services(
    services: InfrastructurePorts,
) -> InfrastructureServices:
    if isinstance(services, InfrastructureServices):
        return services
    return build_infrastructure_services(
        environment=services.environment,
        filesystem=services.filesystem,
        processes=services.processes,
    )


@dataclass(slots=True)
class InfrastructureServiceContext(InfrastructureServiceContextPort):
    current: InfrastructureServices = field(default_factory=build_infrastructure_services)

    def get(self) -> InfrastructureServices:
        return self.current

    def set(self, services: InfrastructurePorts) -> InfrastructureServices:
        self.current = _coerce_infrastructure_services(services)
        return self.current

    def reset(self) -> InfrastructureServices:
        self.current = build_infrastructure_services()
        return self.current


_infrastructure_service_context = InfrastructureServiceContext()


def get_infrastructure_services() -> InfrastructureServices:
    return _infrastructure_service_context.get()


def set_infrastructure_services(services: InfrastructurePorts) -> InfrastructureServices:
    return _infrastructure_service_context.set(services)


def reset_infrastructure_services() -> InfrastructureServices:
    return _infrastructure_service_context.reset()