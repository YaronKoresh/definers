from __future__ import annotations

import json
import os
import re


def _ui_error(message: str):
    try:
        import gradio as gr

        return gr.Error(message)
    except Exception:
        return ValueError(message)


def _create_train_activity_reporter(total_steps: int):
    from definers.system.download_activity import create_activity_reporter

    return create_activity_reporter(total_steps)


def normalize_selected_rows(selected_rows):
    from definers.constants import MAX_CONSECUTIVE_SPACES, MAX_INPUT_LENGTH
    from definers.ml import simple_text

    if selected_rows is None:
        return None
    if len(selected_rows) > MAX_INPUT_LENGTH:
        raise _ui_error(
            f"Selected rows input too long ({len(selected_rows)} > {MAX_INPUT_LENGTH})"
        )
    if " " * (MAX_CONSECUTIVE_SPACES + 1) in selected_rows:
        raise _ui_error("Selected rows contains too many consecutive spaces")
    return simple_text(selected_rows)


def _normalize_short_text(name: str, value):
    from definers.constants import MAX_CONSECUTIVE_SPACES, MAX_INPUT_LENGTH

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError(f"{name} too long ({len(text)} > {MAX_INPUT_LENGTH})")
    if " " * (MAX_CONSECUTIVE_SPACES + 1) in text:
        raise ValueError(f"{name} contains too many consecutive spaces")
    return text


def _normalize_output_name(value):
    from definers.system.output_paths import managed_output_path

    output_name = _normalize_short_text("save_as", value)
    if output_name is None:
        return None
    base_name = os.path.basename(output_name)
    if not os.path.splitext(base_name)[1]:
        base_name = f"{base_name}.joblib"
    return managed_output_path(
        section="train",
        filename=base_name,
    )


def _coerce_uploaded_value(value):
    if hasattr(value, "name") and not isinstance(value, (str, bytes)):
        return getattr(value, "name")
    return value


def _parse_json_payload(payload):
    normalized_payload = payload
    if normalized_payload is None:
        return None
    normalized_text = str(normalized_payload).strip()
    if not normalized_text:
        return None
    try:
        return json.loads(normalized_text)
    except json.JSONDecodeError:
        return normalized_text


def _parse_numeric_matrix(payload):
    parsed_payload = _parse_json_payload(payload)
    if parsed_payload is None:
        return None
    if not isinstance(parsed_payload, str):
        return parsed_payload
    rows = []
    for raw_line in parsed_payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        segments = [
            segment for segment in re.split(r"[,;\t ]+", line) if segment
        ]
        rows.append([float(segment) for segment in segments])
    return rows or None


def _parse_vocabulary(payload):
    parsed_payload = _parse_json_payload(payload)
    if parsed_payload is None:
        return None
    if isinstance(parsed_payload, list):
        return [str(item) for item in parsed_payload]
    if isinstance(parsed_payload, tuple):
        return [str(item) for item in parsed_payload]
    return [
        item.strip()
        for item in re.split(r"[\n,;]+", str(parsed_payload))
        if item.strip()
    ]


def _serialize_output(value):
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return str(value)
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {
            str(key): _serialize_output(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_serialize_output(item) for item in value]
    return value


def _json_preview(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(_serialize_output(value), indent=2, ensure_ascii=False)


def _looks_like_artifact_path(value) -> bool:
    if not isinstance(value, str):
        return False
    normalized_value = value.strip().lower()
    if not normalized_value:
        return False
    known_suffixes = (
        ".csv",
        ".flac",
        ".joblib",
        ".jpeg",
        ".jpg",
        ".json",
        ".mkv",
        ".mp3",
        ".mp4",
        ".onnx",
        ".pkl",
        ".png",
        ".pt",
        ".pth",
        ".safetensors",
        ".txt",
        ".wav",
    )
    return normalized_value.endswith(known_suffixes)


def _render_markdown(title: str, status: str, details):
    lines = [f"## {title}", f"- Status: {status}"]
    for detail in details:
        lines.append(f"- {detail}")
    return "\n".join(lines)


def _split_result_output(title: str, result):
    if result is None:
        return (
            None,
            _render_markdown(title, "blocked", ["No result returned."]),
            "",
        )
    if _looks_like_artifact_path(result):
        return (
            result,
            _render_markdown(title, "ready", [f"Artifact: {result}"]),
            result,
        )
    return (
        None,
        _render_markdown(title, "ready", ["Result returned inline."]),
        _json_preview(result),
    )


def _require_value(name: str, value):
    if value is None:
        raise _ui_error(f"{name} is required")
    if isinstance(value, str) and not value.strip():
        raise _ui_error(f"{name} is required")
    return value


def _build_training_request(
    features,
    labels,
    resume_model,
    remote_src,
    dataset_label_columns,
    revision,
    url_type,
    drop_list,
    selected_rows,
    batch_size,
    validation_split,
    test_split,
    order_by,
    stratify,
    save_as=None,
):
    source_type = url_type or "parquet"
    return {
        "data": remote_src or features,
        "target": labels,
        "resume_from": _coerce_uploaded_value(resume_model),
        "save_as": _normalize_output_name(save_as),
        "revision": _normalize_short_text("revision", revision),
        "source_type": source_type,
        "label_columns": dataset_label_columns,
        "drop": drop_list,
        "select": normalize_selected_rows(selected_rows),
        "batch_size": int(batch_size or 32),
        "validation_split": float(validation_split or 0.0),
        "test_split": float(test_split or 0.0),
        "order_by": _normalize_short_text("order_by", order_by),
        "stratify": _normalize_short_text("stratify", stratify),
    }


def _create_trainer(request):
    from definers.ml import AutoTrainer

    return AutoTrainer(
        revision=request["revision"],
        source_type=request["source_type"],
        batch_size=request["batch_size"],
        validation_split=request["validation_split"],
        test_split=request["test_split"],
    )


def build_training_plan_markdown(
    features,
    labels,
    resume_model,
    remote_src,
    dataset_label_columns,
    revision,
    url_type,
    drop_list,
    selected_rows,
    batch_size,
    validation_split,
    test_split,
    order_by,
    stratify,
):
    from definers.ml.trainer_plan import (
        render_training_plan_markdown,
    )

    report = _create_train_activity_reporter(3)
    report(
        1,
        "Normalize training request",
        detail="Collecting the selected training inputs.",
    )
    request = _build_training_request(
        features,
        labels,
        resume_model,
        remote_src,
        dataset_label_columns,
        revision,
        url_type,
        drop_list,
        selected_rows,
        batch_size,
        validation_split,
        test_split,
        order_by,
        stratify,
    )
    report(
        2,
        "Build training plan",
        detail="Building the training execution plan.",
    )
    trainer = _create_trainer(request)
    plan = trainer.training_plan(
        data=request["data"],
        target=request["target"],
        resume_from=request["resume_from"],
        label_columns=request["label_columns"],
        drop=request["drop"],
        select=request["select"],
        order_by=request["order_by"],
        stratify=request["stratify"],
    )
    report(
        3,
        "Render training plan",
        detail="Rendering the training plan preview.",
    )
    return render_training_plan_markdown(plan)


def handle_training(
    features,
    labels,
    resume_model,
    remote_src,
    dataset_label_columns,
    revision,
    url_type,
    drop_list,
    selected_rows,
    save_as,
    batch_size,
    validation_split,
    test_split,
    order_by,
    stratify,
):
    from definers.ml.trainer_plan import (
        render_training_plan_markdown,
    )

    report = _create_train_activity_reporter(5)
    report(
        1,
        "Normalize training request",
        detail="Collecting the selected training inputs.",
    )
    request = _build_training_request(
        features,
        labels,
        resume_model,
        remote_src,
        dataset_label_columns,
        revision,
        url_type,
        drop_list,
        selected_rows,
        batch_size,
        validation_split,
        test_split,
        order_by,
        stratify,
        save_as=save_as,
    )
    report(
        2,
        "Initialize trainer",
        detail="Preparing the training runtime.",
    )
    trainer = _create_trainer(request)
    report(
        3,
        "Build training plan",
        detail="Preparing the training plan snapshot.",
    )
    plan = trainer.training_plan(
        data=request["data"],
        target=request["target"],
        resume_from=request["resume_from"],
        label_columns=request["label_columns"],
        drop=request["drop"],
        select=request["select"],
        order_by=request["order_by"],
        stratify=request["stratify"],
    )
    plan_markdown = render_training_plan_markdown(plan)
    report(
        4,
        "Run training job",
        detail="Running the model training workflow.",
    )
    model_output = trainer.train(
        data=request["data"],
        target=request["target"],
        save_as=request["save_as"],
        revision=request["revision"],
        source_type=request["source_type"],
        label_columns=request["label_columns"],
        drop=request["drop"],
        select=request["select"],
        order_by=request["order_by"],
        stratify=request["stratify"],
        validation_split=request["validation_split"],
        test_split=request["test_split"],
        batch_size=request["batch_size"],
        resume_from=request["resume_from"],
    )
    report(
        5,
        "Render training status",
        detail="Preparing the training status summary.",
    )
    status_markdown = _render_markdown(
        "Training",
        "ready" if model_output else "blocked",
        [
            f"Artifact: {model_output}"
            if model_output
            else "No artifact produced.",
            f"Mode Source: {request['source_type']}",
            f"Batch Size: {request['batch_size']}",
        ],
    )
    return model_output, plan_markdown, status_markdown


def handle_prediction(model_predict, prediction_data, prediction_payload):
    from definers.ml import AutoTrainer

    report = _create_train_activity_reporter(4)
    report(
        1,
        "Validate prediction inputs",
        detail="Checking the prediction model and payload.",
    )
    report(
        2,
        "Load saved model",
        detail="Loading the saved model artifact.",
    )
    trainer = AutoTrainer(model_path=_coerce_uploaded_value(model_predict))
    prediction_source = prediction_data
    if prediction_source is None:
        prediction_source = _parse_json_payload(prediction_payload)
    _require_value("Prediction input", prediction_source)
    report(
        3,
        "Run prediction",
        detail="Running the prediction workflow.",
    )
    result = trainer.predict(prediction_source)
    report(
        4,
        "Render prediction output",
        detail="Preparing the prediction output.",
    )
    return _split_result_output("Prediction", result)


def handle_inference(model_task, inference_data, model_type):
    from definers.ml import AutoTrainer

    report = _create_train_activity_reporter(4)
    report(
        1,
        "Validate inference request",
        detail="Checking the task, input, and model type.",
    )
    task = _require_value(
        "Model task", _normalize_short_text("task", model_task)
    )
    data = _require_value("Inference input", inference_data)
    normalized_model_type = _normalize_short_text("model_type", model_type)
    if normalized_model_type == "auto":
        normalized_model_type = None
    report(
        2,
        "Initialize inference runtime",
        detail="Preparing the inference runtime.",
    )
    trainer = AutoTrainer(task=task)
    report(
        3,
        "Run inference",
        detail="Running the task inference workflow.",
    )
    result = trainer.infer(
        data,
        task=task,
        model_type=normalized_model_type,
    )
    report(
        4,
        "Render inference output",
        detail="Preparing the inference output.",
    )
    return _split_result_output("Inference", result)


def handle_answer(prompt, history_json, attachment):
    from definers.ml import answer, init_pretrained_model

    report = _create_train_activity_reporter(4)
    report(
        1,
        "Normalize answer history",
        detail="Normalizing the prompt history and attachment payload.",
    )
    history = _parse_json_payload(history_json)
    if history is None:
        history = []
    if not isinstance(history, list):
        raise ValueError("History must be a JSON list of messages")
    if attachment is not None:
        history.append(
            {
                "role": "user",
                "content": {"path": _coerce_uploaded_value(attachment)},
            }
        )
    normalized_prompt = prompt.strip() if isinstance(prompt, str) else ""
    if normalized_prompt:
        history.append({"role": "user", "content": normalized_prompt})
    _require_value("Prompt or history", history)
    report(
        2,
        "Prepare answer runtime",
        detail="Initializing the answer model runtime.",
    )
    init_pretrained_model("answer", True)
    report(
        3,
        "Run answer runtime",
        detail="Generating the answer response.",
    )
    response = answer(history)
    report(
        4,
        "Finalize answer output",
        detail="Preparing the answer output.",
    )
    return response


def handle_text_feature_extraction(text):
    from definers.ml import extract_text_features

    report = _create_train_activity_reporter(4)
    report(
        1,
        "Validate source text",
        detail="Checking the source text payload.",
    )
    normalized_text = _require_value("Text", text)
    report(
        2,
        "Extract feature matrix",
        detail="Extracting the text feature matrix.",
    )
    features = extract_text_features(normalized_text)
    report(
        3,
        "Serialize features",
        detail="Serializing the extracted feature matrix.",
    )
    serialized = _serialize_output(features)
    row_count = len(serialized) if isinstance(serialized, list) else 1
    column_count = 0
    if isinstance(serialized, list) and serialized:
        first_row = serialized[0]
        if isinstance(first_row, list):
            column_count = len(first_row)
        else:
            column_count = len(serialized)
            row_count = 1
    report(
        4,
        "Render feature summary",
        detail="Preparing the feature extraction summary.",
    )
    summary = _render_markdown(
        "Text Features",
        "ready",
        [f"Rows: {row_count}", f"Columns: {column_count}"],
    )
    return _json_preview(serialized), summary


def handle_features_to_text(feature_payload, vocabulary_payload):
    from definers.ml import features_to_text

    report = _create_train_activity_reporter(3)
    report(
        1,
        "Validate feature payload",
        detail="Checking the feature payload.",
    )
    features = _require_value(
        "Feature payload", _parse_json_payload(feature_payload)
    )
    report(
        2,
        "Resolve vocabulary",
        detail="Preparing the optional reconstruction vocabulary.",
    )
    vocabulary = _parse_vocabulary(vocabulary_payload)
    report(
        3,
        "Reconstruct text",
        detail="Reconstructing text from the feature payload.",
    )
    return features_to_text(features, vocabulary=vocabulary)


def handle_quick_summary(text):
    from definers.ml import init_pretrained_model, summarize

    report = _create_train_activity_reporter(3)
    report(
        1,
        "Validate source text",
        detail="Checking the source text for summarization.",
    )
    normalized_text = _require_value("Text", text)
    report(
        2,
        "Load summary runtime",
        detail="Initializing the summary runtime.",
    )
    init_pretrained_model("summary", True)
    report(
        3,
        "Generate summary",
        detail="Generating the one-pass summary.",
    )
    return summarize(normalized_text)


def handle_map_reduce_summary(text, max_words):
    from definers.ml import init_pretrained_model, map_reduce_summary

    report = _create_train_activity_reporter(3)
    report(
        1,
        "Validate source text",
        detail="Checking the source text for map-reduce summarization.",
    )
    normalized_text = _require_value("Text", text)
    report(
        2,
        "Load summary runtime",
        detail="Initializing the summary runtime.",
    )
    init_pretrained_model("summary", True)
    report(
        3,
        "Run map-reduce summary",
        detail="Running the map-reduce summarization flow.",
    )
    return map_reduce_summary(normalized_text, int(max_words))


def handle_iterative_summary(text, max_words, min_loops):
    from definers.ml import init_pretrained_model, summary

    report = _create_train_activity_reporter(3)
    report(
        1,
        "Validate source text",
        detail="Checking the source text for iterative summarization.",
    )
    normalized_text = _require_value("Text", text)
    report(
        2,
        "Load summary runtime",
        detail="Initializing the summary runtime.",
    )
    init_pretrained_model("summary", True)
    report(
        3,
        "Run iterative summary",
        detail="Running the iterative summarization flow.",
    )
    return summary(
        normalized_text,
        max_words=int(max_words),
        min_loops=int(min_loops),
    )


def handle_prompt_optimization(prompt):
    from definers.constants import general_positive_prompt
    from definers.ml import (
        init_pretrained_model,
        preprocess_prompt,
    )

    report = _create_train_activity_reporter(5)
    report(
        1,
        "Validate prompt",
        detail="Checking the prompt payload.",
    )
    normalized_prompt = _require_value("Prompt", prompt)
    report(
        2,
        "Load summary runtime",
        detail="Initializing the summary runtime.",
    )
    init_pretrained_model("summary", True)
    report(
        3,
        "Load translation runtime",
        detail="Initializing the translation runtime.",
    )
    init_pretrained_model("translate", True)
    report(
        4,
        "Preprocess prompt",
        detail="Preprocessing the prompt payload.",
    )
    preprocessed_prompt = preprocess_prompt(normalized_prompt)
    report(
        5,
        "Optimize prompt",
        detail="Producing the optimized prompt variant.",
    )
    optimized_prompt = (
        f"{preprocessed_prompt}, {general_positive_prompt}, "
        f"{general_positive_prompt}."
    )
    return preprocessed_prompt, optimized_prompt


def handle_ml_health_report():
    from definers.ml import get_ml_health_snapshot, ml_health_markdown

    report = _create_train_activity_reporter(3)
    report(
        1,
        "Collect health snapshot",
        detail="Collecting the live ML health snapshot.",
    )
    snapshot = get_ml_health_snapshot()
    report(
        2,
        "Render health report",
        detail="Preparing the ML health report.",
    )
    status = _render_markdown(
        "Validation",
        "ready",
        [
            f"Training Ready: {snapshot.training_ready}",
            f"Answer Runtime Ready: {snapshot.answer_runtime_ready}",
        ],
    )
    report(
        3,
        "Finalize health output",
        detail="Preparing the validation status output.",
    )
    return ml_health_markdown(), status


def handle_validate_ml_health():
    from definers.ml import validate_ml_health

    report = _create_train_activity_reporter(3)
    report(
        1,
        "Validate runtime",
        detail="Validating the ML runtime dependencies.",
    )
    try:
        report(
            2,
            "Inspect dependencies",
            detail="Inspecting the runtime dependencies.",
        )
        validate_ml_health()
    except Exception as error:
        report(
            3,
            "Finalize validation output",
            detail="Preparing the validation failure details.",
        )
        return _render_markdown("Validation", "blocked", [str(error)])
    report(
        3,
        "Finalize validation output",
        detail="Preparing the validation success summary.",
    )
    return _render_markdown(
        "Validation",
        "ready",
        ["Training and data preparation capabilities are available."],
    )


def handle_kmeans_suggestions(feature_matrix, k_min, k_max, random_state):
    from definers.ml import kmeans_k_suggestions

    report = _create_train_activity_reporter(4)
    report(
        1,
        "Validate feature matrix",
        detail="Checking the feature matrix payload.",
    )
    matrix = _require_value(
        "Feature matrix", _parse_numeric_matrix(feature_matrix)
    )
    start_k = int(k_min)
    end_k = int(k_max)
    if end_k <= start_k:
        raise ValueError("k_max must be greater than k_min")
    report(
        2,
        "Resolve k range",
        detail=f"Scoring cluster counts from {start_k} to {end_k}.",
    )
    report(
        3,
        "Score cluster counts",
        detail="Evaluating the candidate cluster counts.",
    )
    result = kmeans_k_suggestions(
        matrix,
        k_range=range(start_k, end_k + 1),
        random_state=None if random_state is None else int(random_state),
    )
    report(
        4,
        "Render k-means metrics",
        detail="Preparing the k-means metrics and suggestions.",
    )
    summary = _render_markdown(
        "K-Means Advisor",
        "ready",
        [
            f"Elbow Suggestion: {result.get('suggested_k_elbow')}",
            f"Silhouette Suggestion: {result.get('suggested_k_silhouette')}",
            f"Final Suggestion: {result.get('final_suggestion')}",
        ],
    )
    return summary, _json_preview(result)


def handle_rvc_checkpoint_lookup(folder_path, model_name):
    from definers.ml import find_latest_rvc_checkpoint

    report = _create_train_activity_reporter(3)
    report(
        1,
        "Validate checkpoint request",
        detail="Checking the checkpoint folder and model name.",
    )
    folder = _require_value(
        "Checkpoint folder", _normalize_short_text("folder_path", folder_path)
    )
    name = _require_value(
        "Model name", _normalize_short_text("model_name", model_name)
    )
    report(
        2,
        "Resolve latest checkpoint",
        detail="Searching for the latest matching checkpoint.",
    )
    checkpoint = find_latest_rvc_checkpoint(folder, name)
    report(
        3,
        "Render checkpoint status",
        detail="Preparing the checkpoint lookup result.",
    )
    if checkpoint is None:
        return _render_markdown(
            "RVC Checkpoint",
            "blocked",
            ["No checkpoint matched the requested pattern."],
        )
    return _render_markdown(
        "RVC Checkpoint",
        "ready",
        [f"Latest Checkpoint: {checkpoint}"],
    )


def handle_language_lookup(language_code):
    from definers.ml import lang_code_to_name

    report = _create_train_activity_reporter(3)
    report(
        1,
        "Validate language code",
        detail="Checking the language code input.",
    )
    code = _require_value(
        "Language code",
        _normalize_short_text("language_code", language_code),
    )
    report(
        2,
        "Resolve language name",
        detail="Resolving the language code to its name.",
    )
    resolved_name = lang_code_to_name(code)
    report(
        3,
        "Render language status",
        detail="Preparing the language lookup result.",
    )
    return _render_markdown(
        "Language Lookup",
        "ready",
        [f"{code} -> {resolved_name}"],
    )


def handle_init_model_files(task, turbo, model_type):
    from definers.ml import init_model_file

    report = _create_train_activity_reporter(4)
    report(
        1,
        "Validate bootstrap config",
        detail="Checking the task and model type.",
    )
    normalized_task = _require_value(
        "Task", _normalize_short_text("task", task)
    )
    normalized_model_type = _normalize_short_text("model_type", model_type)
    if normalized_model_type == "auto":
        normalized_model_type = None
    report(
        2,
        "Resolve model type",
        detail="Resolving the requested model type.",
    )
    report(
        3,
        "Initialize model files",
        detail="Initializing the model file artifacts.",
    )
    result = init_model_file(
        normalized_task,
        turbo=bool(turbo),
        model_type=normalized_model_type,
    )
    details = [f"Task: {normalized_task}"]
    if result:
        details.append(f"Artifact: {result}")
    report(
        4,
        "Render model file status",
        detail="Preparing the model file initialization result.",
    )
    return _render_markdown("Model File Init", "ready", details)


def handle_load_runtime_model(task, turbo):
    from definers.ml import init_pretrained_model

    report = _create_train_activity_reporter(3)
    report(
        1,
        "Validate bootstrap config",
        detail="Checking the runtime model request.",
    )
    normalized_task = _require_value(
        "Task", _normalize_short_text("task", task)
    )
    report(
        2,
        "Load runtime model",
        detail="Loading the runtime model.",
    )
    init_pretrained_model(normalized_task, bool(turbo))
    report(
        3,
        "Render runtime status",
        detail="Preparing the runtime model status.",
    )
    return _render_markdown(
        "Runtime Model Init",
        "ready",
        [f"Task: {normalized_task}", "Model loaded into runtime."],
    )


__all__ = [
    "build_training_plan_markdown",
    "handle_answer",
    "handle_features_to_text",
    "handle_inference",
    "handle_init_model_files",
    "handle_iterative_summary",
    "handle_kmeans_suggestions",
    "handle_language_lookup",
    "handle_load_runtime_model",
    "handle_map_reduce_summary",
    "handle_ml_health_report",
    "handle_prediction",
    "handle_prompt_optimization",
    "handle_quick_summary",
    "handle_rvc_checkpoint_lookup",
    "handle_text_feature_extraction",
    "handle_training",
    "handle_validate_ml_health",
    "normalize_selected_rows",
]
