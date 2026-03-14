import os

from definers.platform.compute import (
    cuda_toolkit as _cuda_toolkit,
    cuda_version,
    device,
    free as _free,
    set_cuda_env as _set_cuda_env,
)
from definers.system import (
    catch,
    directory,
    get_os_name,
    log,
    paths,
    permit,
    run,
)


def cuda_toolkit():
    return _cuda_toolkit(
        directory_func=directory,
        permit_func=permit,
        run_func=run,
    )


def set_cuda_env():
    return _set_cuda_env(
        get_os_name_func=get_os_name,
        paths_func=paths,
        log_func=log,
        environ=os.environ,
    )


def free():
    return _free(catch_func=catch, run_func=run, environ=os.environ)


__all__ = ("cuda_toolkit", "cuda_version", "device", "free", "set_cuda_env")
