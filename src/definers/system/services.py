from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

import definers.system.filesystem as _filesystem
import definers.system.processes as _processes
import definers.system.runtime as _runtime
from definers.system.contracts import (
    CommandInput,
    EnvironmentPort,
    FileSystemPort,
    InfrastructurePorts,
    InfrastructureServiceContextPort,
    PathInput,
    ProcessEnvironment,
    ProcessPort,
)


@dataclass(slots=True)
class EnvironmentService:
    get_os_name: Callable[[], str] = _runtime.get_os_name
    is_admin_windows: Callable[[], bool] = _runtime.is_admin_windows
    cores: Callable[[], int | None] = _runtime.cores
    get_python_version: Callable[[], str | None] = _runtime.get_python_version
    importable: Callable[[str], bool] = _runtime.importable
    runnable: Callable[[str], bool] = _runtime.runnable
    check_version_wildcard: Callable[[Any, Any], bool] = (
        _runtime.check_version_wildcard
    )
    installed: Callable[[str, str | None], bool] = _runtime.installed


@dataclass(slots=True)
class FileSystemService:
    exist: Callable[..., bool] = _filesystem.exist
    load: Callable[[str], Any] = _filesystem.load
    read: Callable[[str], Any] = _filesystem.read
    save: Callable[[str, Any], Any] = _filesystem.save
    write: Callable[[str, Any], Any] = _filesystem.write
    directory: Callable[[str, bool], Any] = _filesystem.directory
    copy: Callable[[str, str], Any] = _filesystem.copy
    move: Callable[[str, str], Any] = _filesystem.move
    delete: Callable[[PathInput], Any] = _filesystem.delete
    remove: Callable[[PathInput], Any] = _filesystem.remove
    permit: Callable[..., bool] = _filesystem.permit


@dataclass(slots=True)
class ProcessService:
    secure_command: Callable[[CommandInput], list[str]] = (
        _processes.secure_command
    )
    run_linux: Callable[[CommandInput, bool, ProcessEnvironment], Any] = (
        _processes.run_linux
    )
    run_windows: Callable[[CommandInput, bool, ProcessEnvironment], Any] = (
        _processes.run_windows
    )
    run: Callable[[CommandInput, bool, ProcessEnvironment], Any] = (
        _processes.run
    )
    get_process_pid: Callable[[str], int | None] = _processes.get_process_pid
    send_signal_to_process: Callable[[int, int], bool] = (
        _processes.send_signal_to_process
    )


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
        get_os_name=environment.get_os_name,
        is_admin_windows=environment.is_admin_windows,
        cores=environment.cores,
        get_python_version=environment.get_python_version,
        importable=environment.importable,
        runnable=environment.runnable,
        check_version_wildcard=environment.check_version_wildcard,
        installed=environment.installed,
    )


def _coerce_filesystem_service(
    filesystem: FileSystemPort | None,
) -> FileSystemService:
    if filesystem is None:
        return FileSystemService()
    if isinstance(filesystem, FileSystemService):
        return filesystem
    return FileSystemService(
        exist=filesystem.exist,
        load=filesystem.load,
        read=filesystem.read,
        save=filesystem.save,
        write=filesystem.write,
        directory=filesystem.directory,
        copy=filesystem.copy,
        move=filesystem.move,
        delete=filesystem.delete,
        remove=filesystem.remove,
        permit=filesystem.permit,
    )


def _coerce_process_service(processes: ProcessPort | None) -> ProcessService:
    if processes is None:
        return ProcessService()
    if isinstance(processes, ProcessService):
        return processes
    return ProcessService(
        secure_command=processes.secure_command,
        run_linux=processes.run_linux,
        run_windows=processes.run_windows,
        run=processes.run,
        get_process_pid=processes.get_process_pid,
        send_signal_to_process=processes.send_signal_to_process,
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
    current: InfrastructureServices = field(
        default_factory=build_infrastructure_services
    )
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def get(self) -> InfrastructureServices:
        with self._lock:
            return self.current

    def set(self, services: InfrastructurePorts) -> InfrastructureServices:
        with self._lock:
            self.current = _coerce_infrastructure_services(services)
            return self.current

    def reset(self) -> InfrastructureServices:
        with self._lock:
            self.current = build_infrastructure_services()
            return self.current


_infrastructure_service_context = InfrastructureServiceContext()


def get_infrastructure_services() -> InfrastructureServices:
    return _infrastructure_service_context.get()


def set_infrastructure_services(
    services: InfrastructurePorts,
) -> InfrastructureServices:
    return _infrastructure_service_context.set(services)


def reset_infrastructure_services() -> InfrastructureServices:
    return _infrastructure_service_context.reset()
