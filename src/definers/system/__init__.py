import logging
import subprocess

from definers import (
    file_ops as _file_ops,
    os_utils as _os_utils,
)
from definers.observability import init_logger as _shared_init_logger

from . import filesystem as _filesystem, paths as _paths
from .archives import (
    compress,
    extract,
    get_ext,
    is_ai_model,
    secure_path,
)
from .installation import (
    apt_install,
    build_faiss,
    install_audio_effects,
    install_ffmpeg,
    install_ffmpeg_linux,
    install_ffmpeg_windows,
    modify_wheel_requirements,
    post_install,
    pre_install,
)
from .services import get_infrastructure_services
from .threads import big_number, thread, wait


def init_logger(
    level: str | int | None = None,
    log_file: str | None = None,
    *,
    enable_console: bool = True,
) -> logging.Logger:
    return _shared_init_logger(
        "definers",
        level=level,
        log_file=log_file,
        enable_console=enable_console,
        propagate=False,
        default_level=logging.INFO,
    )


logger = init_logger()


def _bind_module_attributes(function, module):
    for attribute_name in dir(module):
        if attribute_name.startswith("__") or hasattr(function, attribute_name):
            continue
        setattr(function, attribute_name, getattr(module, attribute_name))
    return function


normalize_path = _paths.normalize_path
full_path = _paths.full_path
paths = _bind_module_attributes(_paths.paths, _paths)
unique = _paths.unique
cwd = _paths.cwd
parent_directory = _paths.parent_directory
path_end = _paths.path_end
path_ext = _paths.path_ext
path_name = _paths.path_name
add_path = _paths.add_path
find_package_paths = _paths.find_package_paths
is_package_path = _paths.is_package_path
tmp = _paths.tmp

log = _file_ops.log
catch = _file_ops.catch
save_temp_text = _file_ops.save_temp_text
is_directory = _filesystem.is_directory
is_symlink = _filesystem.is_symlink
remove_readonly = _filesystem.remove_readonly
shutil_rmtree_readonly_handler = _filesystem.shutil_rmtree_readonly_handler
is_text = _filesystem.is_text

get_linux_distribution = _os_utils.get_linux_distribution


def get_os_name():
    return get_infrastructure_services().environment.get_os_name()


def is_admin_windows():
    return get_infrastructure_services().environment.is_admin_windows()


def cores():
    return get_infrastructure_services().environment.cores()


def check_version_wildcard(version_spec, version_actual):
    return get_infrastructure_services().environment.check_version_wildcard(
        version_spec, version_actual
    )


def installed(package_name: str, version: str | None = None) -> bool:
    return get_infrastructure_services().environment.installed(
        package_name, version
    )


def importable(name: str) -> bool:
    return get_infrastructure_services().environment.importable(name)


def runnable(command: str) -> bool:
    return get_infrastructure_services().environment.runnable(command)


def secure_command(command):
    return get_infrastructure_services().processes.secure_command(command)


def exist(*path_parts: str) -> bool:
    return get_infrastructure_services().filesystem.exist(*path_parts)


def copy(source: str, target: str):
    return get_infrastructure_services().filesystem.copy(source, target)


def directory(path: str, exist_ok: bool = True):
    return get_infrastructure_services().filesystem.directory(path, exist_ok)


def move(source: str, target: str):
    return get_infrastructure_services().filesystem.move(source, target)


def delete(path):
    return get_infrastructure_services().filesystem.delete(path)


def remove(path):
    return get_infrastructure_services().filesystem.remove(path)


def load(path: str):
    return get_infrastructure_services().filesystem.load(path)


def read(path: str):
    return get_infrastructure_services().filesystem.read(path)


def write(path: str, text=""):
    return get_infrastructure_services().filesystem.write(path, text)


def save(path: str, text=""):
    return get_infrastructure_services().filesystem.save(path, text)


def run_linux(command, silent: bool = False, env=None):
    return get_infrastructure_services().processes.run_linux(
        command, silent=silent, env={} if env is None else env
    )


def run_windows(command, silent: bool = False, env=None):
    return get_infrastructure_services().processes.run_windows(
        command, silent=silent, env={} if env is None else env
    )


def run(command, silent: bool = False, env=None):
    return get_infrastructure_services().processes.run(
        command, silent=silent, env={} if env is None else env
    )


def get_process_pid(process_name: str) -> int | None:
    return get_infrastructure_services().processes.get_process_pid(process_name)


def send_signal_to_process(pid: int, signal_number: int) -> bool:
    return get_infrastructure_services().processes.send_signal_to_process(
        pid, signal_number
    )


def permit(
    path: str,
    *,
    exists_func=None,
    get_os_name_func=None,
    subprocess_module=None,
) -> bool:
    return get_infrastructure_services().filesystem.permit(
        path,
        exists_func=exist if exists_func is None else exists_func,
        get_os_name_func=get_os_name
        if get_os_name_func is None
        else get_os_name_func,
        subprocess_module=subprocess
        if subprocess_module is None
        else subprocess_module,
    )


__all__ = (
    "add_path",
    "apt_install",
    "big_number",
    "build_faiss",
    "catch",
    "check_version_wildcard",
    "compress",
    "copy",
    "cores",
    "cwd",
    "delete",
    "directory",
    "exist",
    "extract",
    "find_package_paths",
    "full_path",
    "get_ext",
    "get_linux_distribution",
    "get_os_name",
    "get_process_pid",
    "importable",
    "init_logger",
    "install_audio_effects",
    "install_ffmpeg",
    "install_ffmpeg_linux",
    "install_ffmpeg_windows",
    "installed",
    "is_admin_windows",
    "is_ai_model",
    "is_directory",
    "is_package_path",
    "is_symlink",
    "is_text",
    "load",
    "log",
    "logger",
    "modify_wheel_requirements",
    "move",
    "normalize_path",
    "parent_directory",
    "path_end",
    "path_ext",
    "path_name",
    "paths",
    "permit",
    "post_install",
    "pre_install",
    "read",
    "remove",
    "remove_readonly",
    "runnable",
    "run",
    "run_linux",
    "run_windows",
    "save",
    "save_temp_text",
    "secure_command",
    "secure_path",
    "send_signal_to_process",
    "shutil_rmtree_readonly_handler",
    "thread",
    "tmp",
    "unique",
    "wait",
    "write",
)
