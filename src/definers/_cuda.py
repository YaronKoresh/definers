import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from definers._system import (
    catch,
    get_os_name,
    run,
)


def cuda_toolkit():
    import definers as _d

    _d.directory("/usr/share/keyrings/")
    _d.directory("/etc/modprobe.d/")
    _d.permit("/tmp")
    _d.permit("/usr/bin")
    _d.permit("/usr/lib")
    _d.permit("/usr/local")
    _d.run(["apt-get", "update"])
    _d.run(["apt-get", "install", "-y", "curl"])
    _d.run(
        [
            "bash",
            "-lc",
            '\n        export PATH=/sbin:$PATH\n        apt-get update\n        apt-get purge nvidia-*\n        echo "blacklist nouveau" > /etc/modprobe.d/blacklist-nouveau.conf\n        echo "options nouveau modeset=0" >> /etc/modprobe.d/blacklist-nouveau.conf\n        apt-get install -y --reinstall dkms\n        apt-get install -f\n        curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb > /usr/share/keyrings/cuda.deb\n        cd /usr/share/keyrings/\n        ar vx cuda.deb\n        tar xvf data.tar.xz\n        mv /usr/share/keyrings/usr/share/keyrings/cuda-archive-keyring.gpg /usr/share/keyrings/cuda-archive-keyring.gpg\n        rm -r /usr/share/keyrings/usr/\n        rm -r /usr/share/keyrings/etc/\n        echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /" > /etc/apt/sources.list.d/CUDA.list\n    ',
        ]
    )
    _d.permit("/usr/share/keyrings/cuda-archive-keyring.gpg")
    _d.permit("/etc/apt/sources.list.d/CUDA.list")
    _d.run(
        [
            "bash",
            "-lc",
            "\n        apt-get update\n        apt-get install -y cuda-toolkit\n    ",
        ]
    )


def cuda_version():
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, check=True
        )
        output = result.stdout
        match = re.search("Build cuda_([\\d\\.]+)", output)
        if match:
            cuda_version = match.group(1).strip(".")
            return cuda_version
        else:
            return False
    except Exception:
        return False


def set_cuda_env():
    import definers as _d

    if get_os_name() != "linux":
        return None
    cu_path = _d.paths("/opt/cuda*/", "/usr/local/cuda*/")
    ld_path = _d.paths(
        "/opt/cuda*/lib",
        "/usr/local/cuda*/lib",
        "/opt/cuda*/lib64",
        "/usr/local/cuda*/lib64",
    )
    if len(cu_path) > 0 and len(ld_path) > 0:
        cu = cu_path[0]
        ld = ld_path[0]
        _d.log("CUDA_PATH", cu, status=True)
        _d.log("LD_LIBRARY_PATH", ld, status=True)
        os.environ["CUDA_PATH"] = cu
        os.environ["LD_LIBRARY_PATH"] = ld
        return
    _d.log("Cuda not found", "Failed setting CUDA environment", status=False)
    return


def free():
    import torch

    import definers as _d

    try:
        torch.cuda.empty_cache()
    except Exception as e:
        _d.catch(e)
    hf_home = os.environ.get("HF_HOME")
    cache_dir = (
        Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
    )
    if cache_dir.exists():
        for entry in cache_dir.iterdir():
            try:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            except Exception as exc:
                _d.catch(exc)

    for path_str in ("/data-nvme/zerogpu-offload", "/opt/ml/checkpoints"):
        path = Path(path_str)
        if path.exists():
            try:
                shutil.rmtree(path)
            except Exception as exc:
                _d.catch(exc)

    try:
        _d.run([sys.executable, "-m", "pip", "cache", "purge"], silent=True)
    except Exception as exc:
        _d.catch(exc)
    mamba_path = os.path.expanduser("~/miniconda3/bin/mamba")
    if os.path.exists(mamba_path):
        _d.run([mamba_path, "clean", "--all"], silent=True)


def device():
    from accelerate import Accelerator

    acc = Accelerator()
    return str(acc.device)
