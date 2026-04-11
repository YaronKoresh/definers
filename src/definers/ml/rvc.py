from __future__ import annotations

import sys
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path


def get_logger():
    current_logger = logger
    if current_logger is None:
        from definers.logger import init_logger

        current_logger = init_logger()
    return current_logger


def rvc_package_root() -> str:
    return str(Path(__file__).resolve().parent.parent)


@contextmanager
def rvc_import_root():
    package_root = rvc_package_root()
    inserted = package_root not in sys.path
    if inserted:
        sys.path.insert(0, package_root)
    try:
        yield package_root
    finally:
        if inserted:
            sys.path.remove(package_root)


def import_rvc_symbol(symbol_name: str, *module_names: str):
    last_error = None
    with rvc_import_root():
        for module_name in module_names:
            try:
                module = import_module(module_name)
                return getattr(module, symbol_name)
            except Exception as error:
                last_error = error
    get_logger().error(
        "RVC dependency unavailable for %s via %s: %s",
        symbol_name,
        ", ".join(module_names),
        last_error,
    )
    return None


def find_latest_rvc_checkpoint(folder_path: str, model_name: str) -> str | None:
    import os
    import re

    from definers.system import secure_path

    current_logger = get_logger()
    directory_checker = is_directory
    if directory_checker is None:
        from definers.system import is_directory as directory_checker

    current_logger.info(
        f"Searching for latest checkpoint in '{folder_path}' with model name '{model_name}'"
    )
    try:
        sanitized_folder = secure_path(folder_path)
    except Exception as error:
        current_logger.error(f"Invalid checkpoint folder: {error}")
        return None
    if not directory_checker(sanitized_folder):
        current_logger.error(f"Error: Folder not found at {sanitized_folder}")
        return None
    pattern = re.compile(f"^{re.escape(model_name)}_e(\\d+)_s(\\d+)\\.pth$")
    latest_checkpoint = None
    latest_epoch = -1
    latest_global_step = -1
    try:
        for filename in os.listdir(sanitized_folder):
            match = pattern.match(filename)
            if match is None:
                continue
            epoch = int(match.group(1))
            global_step = int(match.group(2))
            if epoch > latest_epoch or (
                epoch == latest_epoch and global_step > latest_global_step
            ):
                latest_epoch = epoch
                latest_global_step = global_step
                latest_checkpoint = filename
    except Exception as error:
        current_logger.error(
            f"An error occurred while scanning the folder for checkpoints: {error}"
        )
        return None
    if latest_checkpoint is not None:
        current_logger.info(f"Latest checkpoint found: {latest_checkpoint}")
    else:
        current_logger.warning(
            f"No checkpoint found matching the pattern in '{sanitized_folder}'"
        )
    return latest_checkpoint


logger = None
is_directory = None
