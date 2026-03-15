from __future__ import annotations

from typing import Any

from definers.application_ml.contracts import (
    AnswerModelPort,
    AnswerProcessorPort,
    AnswerRuntimePort,
    AnswerServicePort,
    ErrorHandlerPort,
    HistoryMessage,
    LogPort,
    ModelRegistryPort,
    PromptProcessingPort,
    SummaryServicePort,
    TextFeatureExtractionPort,
    TextFeatureVectorizerPort,
    TrainableModelPort,
    TrainingArrayAdapterPort,
    TrainingServicePort,
)


def _normalize_answer_text(text: str, required_lang: str) -> str:
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


def _content_paths(content: object) -> list[str]:
    if isinstance(content, dict):
        return [content["path"]]
    if isinstance(content, tuple):
        return [item["path"] for item in content if isinstance(item, dict)]
    return []


def _read_answer_audio(path: str, soundfile_module: Any, librosa_module: Any):
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


def _read_answer_image(path: str, image_module: Any):
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
                (width, height) = shape[:2]
                (max_width, _max_height) = get_max_resolution(
                    width, height, mega_pixels=0.25
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


def _append_history_message(
    history_items: list[HistoryMessage], role: str, content: str
) -> None:
    stripped_content = content.strip()
    if history_items[-1]["role"] != role:
        history_items.append({"role": role, "content": stripped_content})
        return
    history_items[-1]["content"] += "\n\n" + stripped_content
    history_items[-1]["content"] = history_items[-1]["content"].strip()


def _prepare_answer_history(
    history: list[HistoryMessage],
    runtime: AnswerRuntimePort,
    image_module: Any,
    soundfile_module: Any,
    librosa_module: Any,
) -> tuple[list[HistoryMessage], list[Any], list[Any]]:
    from definers.system import get_ext, read

    required_lang = "en"
    image_items: list[Any] = []
    audio_items: list[Any] = []
    prepared_history: list[HistoryMessage] = [
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
            add_content = _normalize_answer_text(content, required_lang)
        else:
            for path in _content_paths(content):
                extension = get_ext(path)
                if extension in runtime.common_audio_formats:
                    loaded_audio = _read_answer_audio(
                        path, soundfile_module, librosa_module
                    )
                    if loaded_audio is not None:
                        audio_items.append(loaded_audio)
                        add_content += f" <|audio_{len(audio_items)}|>"
                elif extension in runtime.iio_formats:
                    image = _read_answer_image(path, image_module)
                    if image is not None:
                        image_items.append(image)
                        add_content += f" <|image_{len(image_items)}|>"
                else:
                    file_content = read(path)
                    add_content += "\n\n" + _normalize_answer_text(
                        file_content, required_lang
                    )
        _append_history_message(prepared_history, role, add_content)
    return (prepared_history, image_items, audio_items)


def _generate_answer_without_processor(
    model: AnswerModelPort,
    history: list[HistoryMessage],
    image_items: list[Any],
    audio_items: list[Any],
):
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


def _generate_answer_with_processor(
    processor: AnswerProcessorPort,
    model: AnswerModelPort,
    history: list[HistoryMessage],
    image_items: list[Any],
    audio_items: list[Any],
):
    from definers.constants import beam_kwargs
    from definers.cuda import device

    prompt = processor.tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=prompt,
        images=image_items if image_items else None,
        audios=audio_items if audio_items else None,
        return_tensors="pt",
    )
    inputs = inputs.to(device())
    generate_ids = model.generate(
        **inputs, **beam_kwargs, max_length=4096, num_logits_to_keep=1
    )
    output_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def answer(history: list[HistoryMessage], runtime: AnswerRuntimePort) -> Any:
    from PIL import Image

    try:
        import librosa
    except Exception:
        librosa = None
    try:
        import soundfile as sf
    except Exception:
        sf = None

    processor = runtime.PROCESSORS.get("answer")
    model = runtime.MODELS.get("answer")
    if model is None:
        return None
    prepared_history, image_items, audio_items = _prepare_answer_history(
        history,
        runtime,
        Image,
        sf,
        librosa,
    )
    if processor is None:
        return _generate_answer_without_processor(
            model,
            prepared_history,
            image_items,
            audio_items,
        )
    return _generate_answer_with_processor(
        processor,
        model,
        prepared_history,
        image_items,
        audio_items,
    )


def fit(
    model: TrainableModelPort,
    array_adapter: TrainingArrayAdapterPort,
    logger: LogPort,
    error_handler: ErrorHandlerPort,
) -> TrainableModelPort:
    from definers.application_ml.training import fit as _fit

    return _fit(
        model,
        array_adapter=array_adapter,
        logger=logger,
        error_handler=error_handler,
    )


def feed(
    model, X_new, y_new=None, epochs: int = 1, logger=None, concatenate=None
):
    from definers.application_ml.training import feed as _feed

    return _feed(
        model,
        X_new,
        y_new,
        epochs=epochs,
        logger=logger,
        concatenate=concatenate,
    )


def initialize_linear_regression(
    input_dim: int,
    model_path: str,
    *,
    runtime,
    factory,
    logger,
):
    from definers.application_ml.training import (
        initialize_linear_regression as _initialize_linear_regression,
    )

    return _initialize_linear_regression(
        input_dim,
        model_path,
        runtime=runtime,
        factory=factory,
        logger=logger,
    )


def train_linear_regression(
    X,
    y,
    model_path: str,
    *,
    learning_rate: float = 0.01,
    runtime,
):
    from definers.application_ml.training import (
        train_linear_regression as _train_linear_regression,
    )

    return _train_linear_regression(
        X,
        y,
        model_path,
        learning_rate=learning_rate,
        runtime=runtime,
    )


def linear_regression(X, y, learning_rate=0.01, epochs=50):
    from definers.application_ml.training import (
        linear_regression as _linear_regression,
    )

    return _linear_regression(X, y, learning_rate=learning_rate, epochs=epochs)


def extract_text_features(
    text: str | None,
    vectorizer: TextFeatureVectorizerPort | None = None,
):
    from definers.application_ml.inference import (
        extract_text_features as _extract_text_features,
    )

    return _extract_text_features(text, vectorizer)


def features_to_text(predicted_features, vectorizer=None, vocabulary=None):
    from definers.application_ml.inference import (
        features_to_text as _features_to_text,
    )

    return _features_to_text(
        predicted_features,
        vectorizer=vectorizer,
        vocabulary=vocabulary,
    )


def predict_linear_regression(X_new, model_path: str, *, factory):
    from definers.application_ml.inference import (
        predict_linear_regression as _predict_linear_regression,
    )

    return _predict_linear_regression(X_new, model_path, factory=factory)


def init_model_file(
    task: str, turbo: bool = True, model_type: str | None = None
):
    from definers.application_ml.repository_sync import (
        init_model_file as _init_model_file,
    )

    return _init_model_file(task, turbo=turbo, model_type=model_type)


def find_latest_rvc_checkpoint(folder_path: str, model_name: str) -> str | None:
    from definers.application_ml.rvc import (
        find_latest_rvc_checkpoint as _find_latest_rvc_checkpoint,
    )

    return _find_latest_rvc_checkpoint(folder_path, model_name)


def lang_code_to_name(code: str):
    from definers.application_ml.introspection import (
        lang_code_to_name as _lang_code_to_name,
    )

    return _lang_code_to_name(code)


def get_cluster_content(model, cluster_index):
    from definers.application_ml.introspection import (
        get_cluster_content as _get_cluster_content,
    )

    return _get_cluster_content(model, cluster_index)


def is_clusters_model(model) -> bool:
    from definers.application_ml.introspection import (
        is_clusters_model as _is_clusters_model,
    )

    return _is_clusters_model(model)


def summarize(text_to_summarize: str) -> str:
    from definers.application_ml.text_generation import summarize as _summarize

    return _summarize(text_to_summarize)


def map_reduce_summary(text: str, max_words: int) -> str:
    from definers.application_ml.text_generation import (
        map_reduce_summary as _map_reduce_summary,
    )

    return _map_reduce_summary(text, max_words)


def summary(text: str, max_words: int = 20, min_loops: int = 1) -> str:
    from definers.application_ml.text_generation import summary as _summary

    return _summary(text, max_words=max_words, min_loops=min_loops)


def preprocess_prompt(prompt: str) -> str:
    from definers.application_ml.text_generation import (
        preprocess_prompt as _preprocess_prompt,
    )

    return _preprocess_prompt(prompt)


def optimize_prompt_realism(prompt: str) -> str:
    from definers.application_ml.text_generation import (
        optimize_prompt_realism as _optimize_prompt_realism,
    )

    return _optimize_prompt_realism(prompt)


def __getattr__(name: str):
    if name in {"HybridModel", "LinearRegressionTorch"}:
        from definers.application_ml import training

        return getattr(training, name)
    raise AttributeError(name)


__all__ = [
    "AnswerModelPort",
    "AnswerProcessorPort",
    "AnswerServicePort",
    "AnswerRuntimePort",
    "HybridModel",
    "LinearRegressionTorch",
    "ModelRegistryPort",
    "PromptProcessingPort",
    "SummaryServicePort",
    "TextFeatureExtractionPort",
    "features_to_text",
    "TextFeatureVectorizerPort",
    "answer",
    "extract_text_features",
    "feed",
    "find_latest_rvc_checkpoint",
    "fit",
    "get_cluster_content",
    "init_model_file",
    "initialize_linear_regression",
    "is_clusters_model",
    "lang_code_to_name",
    "linear_regression",
    "predict_linear_regression",
    "TrainingServicePort",
    "train_linear_regression",
    "map_reduce_summary",
    "optimize_prompt_realism",
    "preprocess_prompt",
    "summarize",
    "summary",
]
