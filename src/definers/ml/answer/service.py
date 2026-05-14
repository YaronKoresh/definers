from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AnswerRuntime:
    MODELS: object
    PROCESSORS: object


def content_paths(content: object) -> list[str]:
    if isinstance(content, dict):
        path = content.get("path")
        return [] if path is None else [str(path)]
    if isinstance(content, tuple):
        return [
            str(item["path"])
            for item in content
            if isinstance(item, dict) and item.get("path") is not None
        ]
    return []


def normalize_answer_text(text: str, required_lang: str) -> str:
    if required_lang == "en" and text.isascii():
        return text
    from definers.text import ai_translate, language

    try:
        content_lang = language(text)
    except Exception:
        return text
    if content_lang != required_lang:
        try:
            return ai_translate(text, lang=required_lang)
        except Exception:
            return text
    return text


def append_history_message(
    history_items: list[dict[str, str]],
    role: str,
    content: str,
) -> None:
    stripped_content = content.strip()
    if history_items[-1]["role"] != role:
        history_items.append({"role": role, "content": stripped_content})
        return
    history_items[-1]["content"] += "\n\n" + stripped_content
    history_items[-1]["content"] = history_items[-1]["content"].strip()


def load_image_module() -> Any | None:
    try:
        from PIL import Image
    except Exception:
        return None
    return Image


def load_soundfile_module() -> Any | None:
    try:
        import soundfile as soundfile_module
    except Exception:
        return None
    return soundfile_module


def load_librosa_module() -> Any | None:
    try:
        import librosa as librosa_module
    except Exception:
        return None
    return librosa_module


def read_answer_image(path: str, image_module: Any) -> Any | None:
    from definers.image import (
        get_max_resolution,
        image_resolution,
        resize_image,
    )

    try:
        return image_module.open(path)
    except Exception:
        try:
            shape = image_resolution(path)
            if (
                isinstance(shape, tuple)
                and len(shape) >= 2
                and shape[0] > 0
                and shape[1] > 0
            ):
                width, height = shape[:2]
                max_width, _max_height = get_max_resolution(
                    width,
                    height,
                    mega_pixels=0.25,
                )
                if max_width > width:
                    resized = resize_image(path, width, height)
                    if isinstance(resized, tuple):
                        return resized[1]
                    return resized
                return image_module.open(path)
        except Exception:
            return None
    return None


def read_answer_audio(
    path: str,
    soundfile_module: Any | None,
    librosa_module: Any | None,
) -> Any | None:
    from definers.audio import audio_preview

    loaded_audio = None
    if soundfile_module is not None:
        try:
            loaded_audio = soundfile_module.read(path)
        except Exception:
            loaded_audio = None
    if loaded_audio is None and librosa_module is not None:
        try:
            preview = audio_preview(file_path=path, max_duration=16)
            source = preview if preview else path
            loaded_audio = librosa_module.load(source, sr=16000, mono=True)
        except Exception:
            loaded_audio = None
    return loaded_audio


def prepare_answer_history(
    history: list[dict[str, Any]],
    runtime: Any,
    dependency_loader: Any,
) -> tuple[list[dict[str, str]], list[Any], list[Any]]:
    from definers.system import get_ext, read

    required_lang = "en"
    unloaded = object()
    image_module: Any = unloaded
    soundfile_module: Any = unloaded
    librosa_module: Any = unloaded
    image_items: list[Any] = []
    audio_items: list[Any] = []
    prepared_history: list[dict[str, str]] = [
        {"role": "system", "content": runtime.SYSTEM_MESSAGE}
    ]
    for message in history:
        content = message["content"]
        role = message["role"]
        add_content = ""
        is_text = not isinstance(content, dict) and not isinstance(
            content, tuple
        )
        if is_text:
            add_content = normalize_answer_text(str(content), required_lang)
        else:
            for path in content_paths(content):
                extension = get_ext(path)
                if extension in runtime.common_audio_formats:
                    if soundfile_module is unloaded:
                        soundfile_module = (
                            dependency_loader.load_soundfile_module()
                        )
                    loaded_audio = read_answer_audio(
                        path, soundfile_module, None
                    )
                    if loaded_audio is None:
                        if librosa_module is unloaded:
                            librosa_module = (
                                dependency_loader.load_librosa_module()
                            )
                        loaded_audio = read_answer_audio(
                            path, None, librosa_module
                        )
                    if loaded_audio is not None:
                        audio_items.append(loaded_audio)
                        add_content += f" <|audio_{len(audio_items)}|>"
                elif extension in runtime.iio_formats:
                    if image_module is unloaded:
                        image_module = dependency_loader.load_image_module()
                    if image_module is not None:
                        image = read_answer_image(path, image_module)
                        if image is not None:
                            image_items.append(image)
                            add_content += f" <|image_{len(image_items)}|>"
                else:
                    try:
                        file_content = read(path)
                    except Exception:
                        continue
                    add_content += "\n\n" + normalize_answer_text(
                        str(file_content),
                        required_lang,
                    )
        if add_content.strip():
            append_history_message(prepared_history, role, add_content)
    return prepared_history, image_items, audio_items


def generate_answer_without_processor(
    model: Any,
    history: list[dict[str, str]],
    image_items: list[Any],
    audio_items: list[Any],
) -> Any:
    prompt = (
        "".join(
            [
                f"<|{message['role']}|>{message['content']}<|end|>"
                for message in history
            ]
        )
        + "<|assistant|>"
    )
    generate_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "max_length": 200,
        "beam_width": 16,
    }
    if image_items:
        generate_kwargs["images"] = image_items
    if audio_items:
        generate_kwargs["audios"] = audio_items
    return model.generate(**generate_kwargs)


def generate_answer_with_processor(
    processor: Any,
    model: Any,
    history: list[dict[str, str]],
    image_items: list[Any],
    audio_items: list[Any],
) -> Any:
    from definers.constants import beam_kwargs
    from definers.cuda import device

    prompt = processor.tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=prompt,
        images=image_items if image_items else None,
        audios=audio_items if audio_items else None,
        return_tensors="pt",
    )
    inputs = inputs.to(device())
    generate_ids = model.generate(
        **inputs,
        **beam_kwargs,
        max_length=4096,
        num_logits_to_keep=1,
    )
    output_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def answer(
    history: list[dict[str, Any]],
    runtime: Any | None = None,
    dependency_loader=None,
):
    if runtime is None:
        from definers.constants import MODELS, PROCESSORS

        runtime = AnswerRuntime(MODELS=MODELS, PROCESSORS=PROCESSORS)
    processor = runtime.PROCESSORS.get("answer")
    model = runtime.MODELS.get("answer")
    if model is None:
        from definers.ml import init_pretrained_model

        init_pretrained_model("answer")
        processor = runtime.PROCESSORS.get("answer")
        model = runtime.MODELS.get("answer")
    if model is None:
        return None
    if dependency_loader is None:
        dependency_loader = sys.modules[__name__]
    prepared_history, image_items, audio_items = prepare_answer_history(
        history,
        runtime,
        dependency_loader,
    )
    if processor is None:
        return generate_answer_without_processor(
            model,
            prepared_history,
            image_items,
            audio_items,
        )
    return generate_answer_with_processor(
        processor,
        model,
        prepared_history,
        image_items,
        audio_items,
    )


__all__ = [glb for glb in globals() if not glb.startswith("_")]
