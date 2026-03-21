from __future__ import annotations

from typing import Any

from definers.constants import MODELS, tasks
from definers.cuda import device, free
from definers.logger import init_logger
from definers.system import get_ext, tmp
from definers.web import download_file

logger = init_logger()


def init_model_file(
    task: str, turbo: bool = True, model_type: str | None = None
):
    from safetensors.torch import load_file

    from definers.system import secure_path

    free()
    model_path = tasks.get(task, task)
    active_model_type = (model_type or get_ext(model_path)).lower()
    if model_path.startswith("http://") or model_path.startswith("https://"):
        temp_model_path = tmp(active_model_type, keep=False)
        model_path = download_file(model_path, temp_model_path)
    try:
        model_path = secure_path(model_path)
        model = _load_model(model_path, active_model_type)
        if model is None:
            return None
        if turbo:
            _apply_turbo_optimizations(model)
        logger.info("✅ Model loaded successfully.")
        MODELS[task] = model
    except FileNotFoundError:
        logger.error(f"Error: The file was not found at '{model_path}'")
        return None
    except Exception as error:
        logger.error(
            f"An unexpected error occurred while loading the model: {error}"
        )
        return None
    finally:
        free()


def _load_model(model_path: str, model_type: str) -> Any:
    supported_types = ["onnx", "pkl", "pt", "pth", "safetensors", "joblib"]

    if model_type not in supported_types:
        logger.error(
            f'Error: Model type "{model_type}" is not supported. Must be one of {supported_types}'
        )
        return None

    logger.info(
        f"Attempting to load a {model_type.upper()} model from: {model_path}"
    )

    if model_type == "joblib":
        import joblib

        return joblib.load(model_path)

    if model_type == "onnx":
        import onnxruntime

        return onnxruntime.InferenceSession(model_path)

    if model_type == "pkl":
        import pickle

        with open(model_path, "rb") as file_obj:
            return pickle.load(file_obj)

    if model_type in ["pt", "pth"]:
        import torch

        model = torch.load(model_path, map_location=device(), weights_only=True)

    else:
        from safetensors.torch import load_file

        model = load_file(model_path, device=device())

    if hasattr(model, "eval"):
        model.eval()
        logger.info("Model set to evaluation mode.")

    return model


def _apply_turbo_optimizations(model: Any) -> None:
    for attr_name, args in [
        ("enable_vae_slicing", ()),
        ("enable_vae_tiling", ()),
        ("enable_model_cpu_offload", ()),
        ("enable_sequential_cpu_offload", ()),
        ("enable_attention_slicing", (1,)),
    ]:
        try:
            getattr(model, attr_name)(*args)
        except AttributeError:
            continue
        except Exception as error:
            logger.warning(f"Could not apply optimization {attr_name}: {error}")
