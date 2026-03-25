import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def cuda_toolkit(*, directory_func, permit_func, run_func):
    directory_func("/usr/share/keyrings/")
    directory_func("/etc/modprobe.d/")
    permit_func("/tmp")
    permit_func("/usr/bin")
    permit_func("/usr/lib")
    permit_func("/usr/local")
    run_func(["apt-get", "update"])
    run_func(["apt-get", "install", "-y", "curl"])
    run_func(
        [
            "bash",
            "-lc",
            '\n        export PATH=/sbin:$PATH\n        apt-get update\n        apt-get purge nvidia-*\n        echo "blacklist nouveau" > /etc/modprobe.d/blacklist-nouveau.conf\n        echo "options nouveau modeset=0" >> /etc/modprobe.d/blacklist-nouveau.conf\n        apt-get install -y --reinstall dkms\n        apt-get install -f\n        curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb > /usr/share/keyrings/cuda.deb\n        cd /usr/share/keyrings/\n        ar vx cuda.deb\n        tar xvf data.tar.xz\n        mv /usr/share/keyrings/usr/share/keyrings/cuda-archive-keyring.gpg /usr/share/keyrings/cuda-archive-keyring.gpg\n        rm -r /usr/share/keyrings/usr/\n        rm -r /usr/share/keyrings/etc/\n        echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /" > /etc/apt/sources.list.d/CUDA.list\n    ',
        ]
    )
    permit_func("/usr/share/keyrings/cuda-archive-keyring.gpg")
    permit_func("/etc/apt/sources.list.d/CUDA.list")
    run_func(
        [
            "bash",
            "-lc",
            "\n        apt-get update\n        apt-get install -y cuda-toolkit\n    ",
        ]
    )


def cuda_version() -> str | bool:
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, check=True
        )
        match = re.search("Build cuda_([\\d\\.]+)", result.stdout)
        if not match:
            return False
        return match.group(1).strip(".")
    except Exception:
        return False


def set_cuda_env(*, get_os_name_func, paths_func, log_func, environ):
    if get_os_name_func() != "linux":
        return None
    cuda_paths = paths_func("/opt/cuda*/", "/usr/local/cuda*/")
    library_paths = paths_func(
        "/opt/cuda*/lib",
        "/usr/local/cuda*/lib",
        "/opt/cuda*/lib64",
        "/usr/local/cuda*/lib64",
    )
    if cuda_paths and library_paths:
        cuda_path = cuda_paths[0]
        library_path = library_paths[0]
        log_func("CUDA_PATH", cuda_path, status=True)
        log_func("LD_LIBRARY_PATH", library_path, status=True)
        environ["CUDA_PATH"] = cuda_path
        environ["LD_LIBRARY_PATH"] = library_path
        return None
    log_func("Cuda not found", "Failed setting CUDA environment", status=False)
    return None


def free(*, catch_func, run_func, environ):
    import torch

    try:
        torch.cuda.empty_cache()
    except Exception as error:
        catch_func(error)

    hf_home = environ.get("HF_HOME")
    cache_dir = (
        Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
    )
    if cache_dir.exists() and _is_ephemeral_cache_dir(cache_dir):
        for entry in cache_dir.iterdir():
            try:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            except Exception as error:
                catch_func(error)

    for path_str in ("/data-nvme/zerogpu-offload", "/opt/ml/checkpoints"):
        path = Path(path_str)
        if not path.exists():
            continue
        try:
            shutil.rmtree(path)
        except Exception as error:
            catch_func(error)

    try:
        run_func([sys.executable, "-m", "pip", "cache", "purge"], silent=True)
    except Exception as error:
        catch_func(error)

    mamba_path = os.path.expanduser("~/miniconda3/bin/mamba")
    if os.path.exists(mamba_path):
        run_func([mamba_path, "clean", "--all"], silent=True)


def _is_ephemeral_cache_dir(cache_dir: Path) -> bool:
    resolved_cache_dir = cache_dir.resolve()
    default_cache_dir = (Path.home() / ".cache" / "huggingface").resolve()
    if resolved_cache_dir == default_cache_dir:
        return False

    ephemeral_roots = [
        Path(tempfile.gettempdir()).resolve(),
        Path("/data-nvme/zerogpu-offload").resolve(),
        Path("/opt/ml/checkpoints").resolve(),
    ]
    return any(
        _is_relative_to(resolved_cache_dir, ephemeral_root)
        for ephemeral_root in ephemeral_roots
    )


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def device() -> str:
    from accelerate import Accelerator

    accelerator = Accelerator()
    return str(accelerator.device)
