from __future__ import annotations

import os
import re

from definers.logger import init_logger
from definers.system import is_directory

logger = init_logger()


def find_latest_rvc_checkpoint(folder_path: str, model_name: str) -> str | None:
    from definers.system import secure_path

    logger.info(
        f"Searching for latest checkpoint in '{folder_path}' with model name '{model_name}'"
    )
    try:
        sanitized_folder = secure_path(folder_path)
    except Exception as error:
        logger.error(f"Invalid checkpoint folder: {error}")
        return None
    if not is_directory(sanitized_folder):
        logger.error(f"Error: Folder not found at {sanitized_folder}")
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
        logger.error(
            f"An error occurred while scanning the folder for checkpoints: {error}"
        )
        return None
    if latest_checkpoint is not None:
        logger.info(f"Latest checkpoint found: {latest_checkpoint}")
    else:
        logger.warning(
            f"No checkpoint found matching the pattern in '{sanitized_folder}'"
        )
    return latest_checkpoint
