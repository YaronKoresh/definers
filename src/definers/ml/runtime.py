from __future__ import annotations

import logging
import random

from definers.constants import MAX_CONSECUTIVE_SPACES, MAX_INPUT_LENGTH


def validate_str_param(name: str, value: str) -> str:
    if value is None:
        return value
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if len(value) > MAX_INPUT_LENGTH:
        raise ValueError(f"{name} too long ({len(value)} > {MAX_INPUT_LENGTH})")
    if " " * (MAX_CONSECUTIVE_SPACES + 1) in value:
        raise ValueError(f"{name} contains too many consecutive spaces")
    return value


def choose_random_words(word_list, num_words=10):
    if not word_list:
        return []
    list_length = len(word_list)
    if num_words > list_length:
        num_words = list_length - 1
    if num_words == 0:
        num_words = 1
    elif num_words == -1:
        return []
    chosen_words = random.sample(word_list, num_words)
    return chosen_words


def pipe(
    task: str,
    *a,
    prompt: str = "",
    path: str = "",
    resolution: str = "640x640",
    length: int = 3,
    fps: int = 24,
):
    import cv2
    import torch
    from diffusers.utils import export_to_video
    from PIL import Image

    from definers import ml as ml_facade
    from definers.constants import MODELS, tasks
    from definers.cuda import device
    from definers.system import big_number, catch, log, tmp

    del a
    del cv2
    try:
        from definers.image import save_image
    except Exception:
        save_image = None

    if MODELS.get(task) is None:
        ml_facade.init_pretrained_model(task)

    params1 = []
    params2 = {}
    if task in ["image", "video"]:
        log("Pipe activated", prompt, status="")
        width, height = resolution.split("x")
        width, height = int(width), int(height)
        if task == "video":
            length = length * fps
        else:
            length = 1
        params2["prompt"] = prompt
        params2["height"] = height
        params2["width"] = width
        params2["guidance_scale"] = 5.0
        if task == "video":
            params2["num_videos_per_prompt"] = 1
            params2["num_frames"] = length
        else:
            params2["max_sequence_length"] = 512
        params2["num_inference_steps"] = 100
        params2["generator"] = torch.Generator(device()).manual_seed(
            random.randint(0, big_number())
        )
    elif task == "detect":
        image = Image.open(path)
        params1.append(image)
    from transformers import AutoTokenizer

    if task in ["detect"]:
        from definers.model_installation import hf_snapshot_download

        local_repo_path = hf_snapshot_download(
            str(tasks[task]),
            item_label=str(tasks[task]),
            detail="Downloading detection model source files.",
        )
        tokenizer = AutoTokenizer.from_pretrained(local_repo_path)
        inputs = tokenizer(*params1, **params2, return_tensors="tf")
    elif task in ["image", "video"]:
        inputs = params2
    try:
        outputs = MODELS[task](**inputs)
    except Exception as error:
        catch(error)
        if task == "image":
            outputs = MODELS["video"](**inputs)
        elif task == "video":
            outputs = MODELS["image"](**inputs)
    if task == "video":
        sample = outputs.frames[0]
        path = tmp("mp4")
        export_to_video(sample, path, fps=24)
        return path
    if task == "image":
        sample = outputs.images[0]
        return save_image(sample)
    if task == "answer":
        return outputs
    if task == "detect":
        preds = {}
        for pred in outputs:
            if pred["label"] not in preds:
                preds[pred["label"]] = []
            preds[pred["label"]].append(pred["box"])
        return preds
    return None


def check_parameter(p):
    return p is not None and (
        not (
            isinstance(p, list)
            and (len(p) == 0 or (isinstance(p[0], str) and p[0].strip() == ""))
            or (isinstance(p, str) and p.strip() == "")
        )
    )


def SklearnWrapper(sklearn_model, is_classification=False):
    import torch

    from definers.runtime_numpy import get_array_module

    np = get_array_module()

    class _SklearnWrapper(torch.nn.Module):
        def __init__(self, sklearn_model, is_classification=False):
            super().__init__()
            self.sklearn_model = sklearn_model
            self.is_classification = is_classification

        def forward(self, x, y=None, y_mask=None):
            del y
            del y_mask
            x_numpy = self._to_numpy(x)
            if (
                hasattr(self.sklearn_model, "predict_proba")
                and self.is_classification
            ):
                predictions = self.sklearn_model.predict_proba(x_numpy)
            elif (
                hasattr(self.sklearn_model, "decision_function")
                and self.is_classification
            ):
                predictions = self.sklearn_model.decision_function(x_numpy)
            else:
                predictions = self.sklearn_model.predict(x_numpy)
            return torch.tensor(
                predictions, dtype=torch.float32, device=x.device
            )

        def fit(self, x, y=None):
            x_numpy = self._to_numpy(x)
            y_numpy = self._to_numpy(y) if y is not None else None
            if y_numpy is not None:
                self.sklearn_model.fit(x_numpy, y_numpy)
            elif len(x_numpy.shape) > 2:
                logging.warning(
                    "Fitting model on 3D input without labels. Fitting on each sequence independently."
                )
                for index in range(x_numpy.shape[0]):
                    self.sklearn_model.fit(x_numpy[index])
            else:
                self.sklearn_model.fit(x_numpy)

        def _to_numpy(self, tensor_or_array):
            if tensor_or_array is None:
                return None
            if isinstance(tensor_or_array, np.ndarray):
                return tensor_or_array
            if isinstance(tensor_or_array, torch.Tensor):
                return tensor_or_array.cpu().numpy()
            raise ValueError(
                f"Expected torch.Tensor or numpy.ndarray, got {type(tensor_or_array)}"
            )

    return _SklearnWrapper(sklearn_model, is_classification)


__all__ = [
    "SklearnWrapper",
    "check_parameter",
    "choose_random_words",
    "pipe",
    "validate_str_param",
]
