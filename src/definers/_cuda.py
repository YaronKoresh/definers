"""CUDA and GPU utilities for the definers package."""

import argparse
import asyncio
import base64
import collections
import collections.abc
import concurrent
import ctypes
import gc
import getpass
import hashlib
import importlib
import inspect
import io
import json
import logging
import math
import multiprocessing
import os
import pathlib
import platform
import queue
import random
import re
import select
import shlex
import shutil
import signal
import site
import string
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import threading
import traceback
import urllib.request
import warnings
import zipfile
from collections import Counter, OrderedDict, namedtuple
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from ctypes.util import find_library
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import lru_cache, partial
from glob import glob
from pathlib import Path
from string import ascii_letters, digits, punctuation
from time import sleep, time
from typing import Any, Optional, Union
from urllib.parse import quote

from definers._system import (
    catch,
    directory,
    get_os_name,
    log,
    paths,
    permit,
    run,
    tmp,
)


def cuda_toolkit():
    import definers as _d

    if get_os_name() != "linux":
        return None

    _d.directory("/usr/share/keyrings/")
    _d.directory("/etc/modprobe.d/")
    _d.permit("/tmp")
    _d.permit("/usr/bin")
    _d.permit("/usr/lib")
    _d.permit("/usr/local")

    _d.run("apt-get update")
    _d.run("apt-get install -y curl")

    _d.run(
        """
        export PATH=/sbin:$PATH
        apt-get update
        apt-get purge nvidia-*
        echo "blacklist nouveau" > /etc/modprobe.d/blacklist-nouveau.conf
        echo "options nouveau modeset=0" >> /etc/modprobe.d/blacklist-nouveau.conf
        apt-get install -y --reinstall dkms
        apt-get install -f
        curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb > /usr/share/keyrings/cuda.deb
        cd /usr/share/keyrings/
        ar vx cuda.deb
        tar xvf data.tar.xz
        mv /usr/share/keyrings/usr/share/keyrings/cuda-archive-keyring.gpg /usr/share/keyrings/cuda-archive-keyring.gpg
        rm -r /usr/share/keyrings/usr/
        rm -r /usr/share/keyrings/etc/
        echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /" > /etc/apt/sources.list.d/CUDA.list
    """
    )

    _d.permit("/usr/share/keyrings/cuda-archive-keyring.gpg")
    _d.permit("/etc/apt/sources.list.d/CUDA.list")

    _d.run(
        """
        apt-get update
        apt-get install -y cuda-toolkit
    """
    )


def cuda_version():
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout
        match = re.search(r"Build cuda_([\d\.]+)", output)
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

    cu_path = _d.paths(
        "/opt/cuda*/",
        "/usr/local/cuda*/",
    )
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

    _d.log(
        "Cuda not found",
        "Failed setting CUDA environment",
        status=False,
    )
    return


def free():
    import torch

    try:
        torch.cuda.empty_cache()
    except Exception as e:
        catch(e)
    run("rm -rf ~/.cache/huggingface/*", silent=True)
    run("rm -rf /data-nvme/zerogpu-offload/*", silent=True)
    run("rm -rf /opt/ml/checkpoints/*", silent=True)
    run("pip cache purge", silent=True)

    mamba_path = os.path.expanduser("~/miniconda3/bin/mamba")
    if os.path.exists(mamba_path):
        run(f"{mamba_path} clean --all", silent=True)


def device():
    from accelerate import Accelerator

    acc = Accelerator()
    return str(acc.device)
