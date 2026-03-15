from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections.abc import Sequence
from typing import Any

import numpy as np

from definers.application_data.contracts import ColumnDatasetPort
from definers.constants import iio_formats


def _catch(error: Exception) -> None:
    try:
        from definers.system import catch as runtime_catch

        runtime_catch(error)
    except Exception:
        logging.error(str(error))


def _delete(path: str | None) -> None:
    if not path:
        return
    try:
        from definers.system import delete as runtime_delete

        runtime_delete(path)
        return
    except Exception:
        return None


def _read(path: str):
    try:
        from definers.system import read as runtime_read

        return runtime_read(path)
    except Exception:
        if os.path.isdir(path):
            return [os.path.join(path, name) for name in os.listdir(path)]
        try:
            with open(path, encoding="utf-8") as file:
                return file.read()
        except OSError:
            return None


def _tmp(extension: str) -> str:
    try:
        from definers.system import tmp as runtime_tmp

        return runtime_tmp(extension)
    except Exception:
        suffix = extension if extension.startswith(".") else f".{extension}"
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        return path


def _runtime():
    import definers.data as data_module

    return data_module


def _has_parameter(values: Sequence[str] | None) -> bool:
    return values is not None and not (
        isinstance(values, list)
        and (
            len(values) == 0
            or (isinstance(values[0], str) and values[0].strip() == "")
        )
    )


def _column_names(dataset: ColumnDatasetPort) -> tuple[str, ...]:
    return tuple(getattr(dataset, "column_names", ()))


def _selected_columns(
    dataset: ColumnDatasetPort,
    requested_columns: Sequence[str],
) -> list[str]:
    requested_names = set(requested_columns)
    return [
        column_name
        for column_name in _column_names(dataset)
        if column_name in requested_names
    ]


def _dataset_slice(
    dataset: ColumnDatasetPort,
    start_index: int,
    end_index: int,
) -> dict[str, Any]:
    return {
        column_name: dataset[column_name][start_index:end_index]
        for column_name in _column_names(dataset)
    }


def _batched_column_values(
    data: dict[str, Any],
    label_names: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    features_batch: list[dict[str, Any]] = []
    labels_batch: list[dict[str, Any]] = []
    batch_size = 0
    for value in data.values():
        if isinstance(value, (list, np.ndarray)):
            batch_size = len(value)
            break
    if batch_size == 0:
        return [], []
    for index in range(batch_size):
        feature_row: dict[str, Any] = {}
        label_row: dict[str, Any] = {}
        for key, value in data.items():
            if key in label_names:
                label_row[key] = value[index]
            else:
                feature_row[key] = value[index]
        features_batch.append(feature_row)
        labels_batch.append(label_row)
    return features_batch, labels_batch


def _path_extension(path: str) -> str | None:
    parts = path.split(".")
    if len(parts) < 2:
        return None
    return parts[-1].strip().lower()


def _load_audio_values(path: str, training: bool):
    import definers

    transformer = definers.sox.Transformer()
    transformer.rate(32000)
    if training:
        temp_wav_path = _tmp("wav")
        transformer.build_file(path, temp_wav_path)
        temp_mp3_path = _tmp("mp3")
        definers.remove_silence(temp_wav_path, temp_mp3_path)
        directory_path, _ = definers.split_mp3(temp_mp3_path, 5)
        files = _read(directory_path) or [temp_mp3_path]
        values = [
            definers.numpy_to_cupy(definers.extract_audio_features(file_path))
            for file_path in files
        ]
        _delete(temp_wav_path)
        _delete(temp_mp3_path)
        _delete(directory_path)
        return values
    temp_mp3_path = _tmp("mp3")
    transformer.build_file(path, temp_mp3_path)
    values = definers.numpy_to_cupy(
        definers.extract_audio_features(temp_mp3_path)
    )
    _delete(temp_mp3_path)
    return values


def _load_table_values(path: str, extension: str):
    import pandas

    if extension == "csv":
        dataframe = pandas.read_csv(path)
        if dataframe.empty:
            dataframe = pandas.read_csv(path, header=None)
    elif extension == "xlsx":
        dataframe = pandas.read_excel(path)
    else:
        dataframe = pandas.read_json(path)
    return dataframe.values


def _load_text_values(path: str):
    import definers

    text = _read(path)
    return definers.numpy_to_cupy(definers.extract_text_features(text))


def _load_image_values(path: str):
    import definers

    resized = definers.resize_image(path, 1024, 1024)
    resized_path = resized[0] if isinstance(resized, tuple) else resized
    return definers.numpy_to_cupy(definers.extract_image_features(resized_path))


def _load_video_values(path: str):
    import definers

    resized_video_file = definers.resize_video(path, 1024, 1024)
    adjusted_fps_file = definers.convert_video_fps(resized_video_file, 24)
    return definers.numpy_to_cupy(
        definers.extract_video_features(adjusted_fps_file)
    )


def load_as_numpy(path: str, training: bool = False):
    try:
        extension = _path_extension(path)
        if extension is None:
            logging.error("Invalid path format: %s", path)
            return None
        if extension in ["wav", "mp3"]:
            return _load_audio_values(path, training)
        if extension in ["csv", "xlsx", "json"]:
            return _load_table_values(path, extension)
        if extension == "txt":
            return _load_text_values(path)
        if extension in iio_formats:
            return _load_image_values(path)
        return _load_video_values(path)
    except Exception as error:
        _catch(error)
        return None


def _load_remote_dataset(src: str, revision: str | None):
    from datasets import load_dataset

    if revision:
        return load_dataset(src, revision=revision, split="train")
    return load_dataset(src, split="train")


def _load_remote_dataset_fallback(
    src: str,
    url_type: str,
    revision: str | None,
):
    from datasets import load_dataset

    if revision:
        return load_dataset(
            url_type,
            data_files={"train": src},
            revision=revision,
            split="train",
        )
    return load_dataset(url_type, data_files={"train": src}, split="train")


def _load_dataset_attempt(load_dataset_call, source_name: str):
    try:
        return load_dataset_call(), True
    except FileNotFoundError:
        logging.error(f"Dataset {source_name} not found.")
        return None, False
    except ConnectionError:
        logging.error(f"Connection error while loading dataset {source_name}.")
        return None, False
    except Exception as error:
        logging.error(f"Error loading dataset {source_name}: {error}")
        return None, True


def fetch_dataset(
    src: str,
    url_type: str | None = None,
    revision: str | None = None,
):
    dataset, allow_fallback = _load_dataset_attempt(
        lambda: _load_remote_dataset(src, revision),
        src,
    )
    if dataset is not None or url_type is None or not allow_fallback:
        return dataset
    fallback_source = f"{url_type} with data_files {src}"
    fallback_dataset, _ = _load_dataset_attempt(
        lambda: _load_remote_dataset_fallback(src, url_type, revision),
        fallback_source,
    )
    return fallback_dataset


def drop_columns(
    dataset: ColumnDatasetPort,
    drop_list: Sequence[str] | None,
):
    if not _has_parameter(drop_list):
        return dataset
    columns_to_delete = _selected_columns(dataset, drop_list)
    return dataset.remove_columns(columns_to_delete)


def select_columns(
    dataset: ColumnDatasetPort,
    cols: Sequence[str] | None,
):
    if not _has_parameter(cols):
        return dataset
    selected_names = set(cols)
    columns_to_drop = [
        column_name
        for column_name in _column_names(dataset)
        if column_name not in selected_names
    ]
    return drop_columns(dataset, columns_to_drop)


def select_rows(
    dataset: ColumnDatasetPort,
    start_index: int,
    end_index: int,
):
    from datasets import Dataset

    return Dataset.from_dict(_dataset_slice(dataset, start_index, end_index))


def split_columns(data, labels: Sequence[str] | None, is_batch: bool = False):
    if not _has_parameter(labels):
        features, label_values = data
        return features, label_values
    if is_batch:
        return _batched_column_values(data, set(labels))
    features = drop_columns(data, labels)
    label_values = select_columns(data, labels)
    return features, label_values


def _is_string_array(value, numpy_module) -> bool:
    return isinstance(value, numpy_module.ndarray) and (
        numpy_module.issubdtype(value.dtype, numpy_module.str_)
        or numpy_module.issubdtype(value.dtype, numpy_module.object_)
    )


def _loaded_values_have_strings(loaded, numpy_module) -> bool:
    if _is_string_array(loaded, numpy_module):
        return True
    if isinstance(loaded, list):
        return any(
            _is_string_array(item, numpy_module)
            for item in loaded
            if item is not None
        )
    return False


def _append_loaded_values(target: list[Any], loaded, convert_value) -> None:
    if isinstance(loaded, list):
        target.extend(
            convert_value(item) for item in loaded if item is not None
        )
        return
    target.append(convert_value(loaded))


def _collect_loaded_values(paths: Sequence[str], role: str, numpy_module):
    runtime = _runtime()

    values: list[Any] = []
    has_strings = False
    for path in paths:
        loaded = load_as_numpy(path, training=True)
        if loaded is None:
            if role == "feature":
                runtime.logger.exception(f"Error loading feature file: {path}")
            else:
                _catch(Exception(f"Error loading label file: {path}"))
            return None
        has_strings = has_strings or _loaded_values_have_strings(
            loaded, numpy_module
        )
        _append_loaded_values(values, loaded, runtime.cupy_to_numpy)
    return values, has_strings


def _stringify_loaded_values(values: Sequence[Any], numpy_module) -> list[str]:
    stringified: list[str] = []
    for value in values:
        if isinstance(value, numpy_module.ndarray):
            stringified.append(" ".join(value.astype(str).flatten()))
        else:
            stringified.append(str(value))
    return stringified


def _tokenize_loaded_values(values: Sequence[Any], numpy_module):
    runtime = _runtime()

    tokenized_values = runtime.tokenize_and_pad(
        _stringify_loaded_values(values, numpy_module),
        runtime.init_tokenizer(),
    )
    return [runtime.cupy_to_numpy(row) for row in tokenized_values]


def _stack_tensor_rows(
    values: Sequence[Any], max_lengths: Sequence[int], runtime
):
    import torch

    return runtime.convert_tensor_dtype(
        torch.stack(
            [
                torch.tensor(runtime.reshape_numpy(value, lengths=max_lengths))
                for value in values
            ]
        )
    )


def _build_tensor_dataset(
    features: Sequence[Any], labels: Sequence[Any], runtime
):
    from torch.utils.data import TensorDataset

    all_data = list(features) + list(labels)
    if not all_data:
        return None
    max_lengths = runtime.get_max_shapes(*all_data)
    features_tensor = _stack_tensor_rows(features, max_lengths, runtime)
    if labels:
        labels_tensor = _stack_tensor_rows(labels, max_lengths, runtime)
        return TensorDataset(features_tensor, labels_tensor)
    return TensorDataset(features_tensor)


def files_to_dataset(
    features_paths: Sequence[str],
    labels_paths: Sequence[str] | None = None,
):
    runtime = _runtime()

    try:
        feature_batch = _collect_loaded_values(features_paths, "feature", np)
        if feature_batch is None:
            return None
        features, features_have_strings = feature_batch
        labels: list[Any] = []
        labels_have_strings = False
        if labels_paths:
            label_batch = _collect_loaded_values(labels_paths, "label", np)
            if label_batch is None:
                return None
            labels, labels_have_strings = label_batch
    except Exception as error:
        _catch(error)
        return None
    if not features and not labels:
        runtime.logger.warning("No valid data loaded.")
        return None
    if features_have_strings:
        features = _tokenize_loaded_values(features, np)
    if labels_paths and labels_have_strings:
        labels = _tokenize_loaded_values(labels, np)
    try:
        return _build_tensor_dataset(features, labels, runtime)
    except Exception as tensor_error:
        _catch(
            Exception(f"Error creating tensor dataset: {type(tensor_error)}")
        )
        _catch(tensor_error)
        return None


def load_source(
    remote_src: str | None = None,
    features: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    url_type: str | None = None,
    revision: str | None = None,
):
    runtime = _runtime()
    if remote_src:
        fetch = getattr(runtime, "fetch_dataset", fetch_dataset)
        return fetch(remote_src, url_type, revision)
    if features:
        build_dataset = getattr(runtime, "files_to_dataset", files_to_dataset)
        return build_dataset(features, labels)
    return None
