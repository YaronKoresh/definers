import os
import sys

from definers.system import (
    catch,
    directory,
    get_os_name,
    log,
    paths,
    permit,
    run,
)
from definers.system.compute import (
    cuda_toolkit as _cuda_toolkit,
    cuda_version,
    device,
    free as _free,
    set_cuda_env as _set_cuda_env,
)


def _current_port(name: str):
    current_module = sys.modules.get(__name__)
    if current_module is None:
        return globals()[name]
    return getattr(current_module, name)


def cuda_toolkit():
    return _cuda_toolkit(
        directory_func=_current_port("directory"),
        permit_func=_current_port("permit"),
        run_func=_current_port("run"),
    )


def set_cuda_env():
    return _set_cuda_env(
        get_os_name_func=_current_port("get_os_name"),
        paths_func=_current_port("paths"),
        log_func=_current_port("log"),
        environ=os.environ,
    )


def free():
    return _free(
        catch_func=_current_port("catch"),
        run_func=_current_port("run"),
        environ=os.environ,
    )


__all__ = [glb for glb in globals() if not glb.startswith("_")]
