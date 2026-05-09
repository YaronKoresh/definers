from __future__ import annotations

from collections.abc import Callable, Iterable
from types import SimpleNamespace

from definers.logger import init_logger
from definers.runtime_numpy import get_numpy_module

logger = init_logger()
np = get_numpy_module()


def _trusted_roots() -> list[str]:
    import os
    import tempfile

    roots = [os.getcwd(), tempfile.gettempdir()]
    configured_root = os.environ.get("DEFINERS_DATA_ROOT", "").strip()
    if configured_root:
        roots.insert(0, configured_root)
    trusted_roots = []
    for root in roots:
        normalized_root = str(root).strip()
        if not normalized_root:
            continue
        try:
            trusted_roots.append(
                os.path.abspath(os.path.expanduser(normalized_root))
            )
        except Exception:
            continue
    return trusted_roots


def safe_path(path: str | None) -> str | None:
    import os

    from definers.system import secure_path

    if not path:
        return None
    try:
        resolved_path = secure_path(
            str(path).strip(),
            trust=_trusted_roots(),
        )
        if not os.path.exists(resolved_path):
            logger.error("Path does not exist: %s", resolved_path)
            return None
        return resolved_path
    except Exception as error:
        logger.exception("Error validating path %s: %s", path, error)
        return None


def catch(error: Exception) -> None:
    try:
        from definers.system import catch as runtime_catch

        runtime_catch(error)
    except Exception:
        import logging

        logging.error(str(error))


def delete(path: str | None) -> None:
    if not path:
        return
    try:
        from definers.system import delete as runtime_delete

        runtime_delete(path)
    except Exception:
        return


def read(path: str):
    import os

    safe_input_path = safe_path(path)
    if safe_input_path is None:
        return None
    try:
        from definers.system import read as runtime_read

        return runtime_read(safe_input_path)
    except Exception:
        if os.path.isdir(safe_input_path):
            return [
                os.path.join(safe_input_path, name)
                for name in os.listdir(safe_input_path)
            ]
        try:
            with open(safe_input_path, encoding="utf-8") as file:
                return file.read()
        except OSError:
            return None


def tmp(extension: str) -> str:
    try:
        from definers.system import tmp as runtime_tmp

        return runtime_tmp(extension)
    except Exception:
        import os
        import tempfile

        suffix = extension if extension.startswith(".") else f".{extension}"
        file_descriptor, path = tempfile.mkstemp(suffix=suffix)
        os.close(file_descriptor)
        return path


def runtime() -> SimpleNamespace:
    import definers.data.arrays as arrays_module
    import definers.data.tokenization as tokenization_module

    return SimpleNamespace(
        convert_tensor_dtype=arrays_module.convert_tensor_dtype,
        cupy_to_numpy=arrays_module.cupy_to_numpy,
        get_max_shapes=arrays_module.get_max_shapes,
        init_tokenizer=tokenization_module.init_tokenizer,
        logger=logger,
        reshape_numpy=arrays_module.reshape_numpy,
        tokenize_and_pad=tokenization_module.tokenize_and_pad,
    )


def has_parameter(values) -> bool:
    return values is not None and not (
        isinstance(values, list)
        and (
            len(values) == 0
            or (isinstance(values[0], str) and values[0].strip() == "")
        )
    )


def column_names(dataset) -> tuple[str, ...]:
    return tuple(getattr(dataset, "column_names", ()))


def selected_columns(dataset, requested_columns) -> list[str]:
    requested_names = set(requested_columns)
    return [
        column_name
        for column_name in column_names(dataset)
        if column_name in requested_names
    ]


def dataset_slice(
    dataset, start_index: int, end_index: int
) -> dict[str, object]:
    return {
        column_name: dataset[column_name][start_index:end_index]
        for column_name in column_names(dataset)
    }


def batched_column_values(
    data, label_names: set[str]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    features_batch = []
    labels_batch = []
    batch_size = 0
    for value in data.values():
        if isinstance(value, (list, np.ndarray)):
            batch_size = len(value)
            break
    if batch_size == 0:
        return [], []
    for index in range(batch_size):
        feature_row = {}
        label_row = {}
        for key, value in data.items():
            if key in label_names:
                label_row[key] = value[index]
            else:
                feature_row[key] = value[index]
        features_batch.append(feature_row)
        labels_batch.append(label_row)
    return features_batch, labels_batch


def path_extension(path: str) -> str | None:
    parts = path.split(".")
    if len(parts) < 2:
        return None
    return parts[-1].strip().lower()


def drop_columns(dataset, drop_list):
    if not has_parameter(drop_list):
        return dataset
    columns_to_delete = selected_columns(dataset, drop_list)
    return dataset.remove_columns(columns_to_delete)


def select_columns(dataset, cols):
    if not has_parameter(cols):
        return dataset
    selected_names = set(cols)
    columns_to_drop = [
        column_name
        for column_name in column_names(dataset)
        if column_name not in selected_names
    ]
    return drop_columns(dataset, columns_to_drop)


def select_rows(dataset, start_index: int, end_index: int):
    from datasets import Dataset

    return Dataset.from_dict(dataset_slice(dataset, start_index, end_index))


def split_columns(data, labels, is_batch: bool = False):
    if not has_parameter(labels):
        features, label_values = data
        return features, label_values
    if is_batch:
        return batched_column_values(data, set(labels))
    features = drop_columns(data, labels)
    label_values = select_columns(data, labels)
    return features, label_values


def ensure_xlsx_runtime() -> None:
    from definers import optional_dependencies

    optional_dependencies.ensure_module_runtime("openpyxl")


def load_audio_values(path: str, training: bool):
    import definers

    transformer = definers.sox.Transformer()
    transformer.rate(32000)
    temp_wav_path = None
    temp_mp3_path = None
    directory_path = None
    try:
        if training:
            temp_wav_path = tmp("wav")
            transformer.build_file(path, temp_wav_path)
            temp_mp3_path = tmp("mp3")
            definers.remove_silence(temp_wav_path, temp_mp3_path)
            directory_path, _ = definers.split_mp3(temp_mp3_path, 5)
            files = read(directory_path) or [temp_mp3_path]
            return [
                definers.numpy_to_cupy(
                    definers.extract_audio_features(file_path)
                )
                for file_path in files
            ]
        temp_mp3_path = tmp("mp3")
        transformer.build_file(path, temp_mp3_path)
        return definers.numpy_to_cupy(
            definers.extract_audio_features(temp_mp3_path)
        )
    finally:
        delete(temp_wav_path)
        delete(temp_mp3_path)
        delete(directory_path)


def load_table_values(path: str, extension: str):
    import pandas

    if extension == "csv":
        dataframe = pandas.read_csv(path)
        if dataframe.empty:
            dataframe = pandas.read_csv(path, header=None)
    elif extension == "xlsx":
        ensure_xlsx_runtime()
        dataframe = pandas.read_excel(path)
    else:
        dataframe = pandas.read_json(path)
    return dataframe.values


def load_text_values(path: str):
    import definers

    text = read(path)
    return definers.numpy_to_cupy(definers.extract_text_features(text))


def load_image_values(path: str):
    import definers

    resized = definers.resize_image(path, 1024, 1024)
    resized_path = resized[0] if isinstance(resized, tuple) else resized
    return definers.numpy_to_cupy(definers.extract_image_features(resized_path))


def load_video_values(path: str):
    import definers

    resized_video_file = definers.resize_video(path, 1024, 1024)
    adjusted_fps_file = definers.convert_video_fps(resized_video_file, 24)
    return definers.numpy_to_cupy(
        definers.extract_video_features(adjusted_fps_file)
    )


def load_as_numpy(path: str, training: bool = False):
    import logging

    from definers.constants import iio_formats

    try:
        safe_input_path = safe_path(path)
        if safe_input_path is None:
            logging.error("Rejected unsafe or invalid path: %s", path)
            return None
        extension = path_extension(safe_input_path)
        if extension is None:
            logging.error("Invalid path format: %s", safe_input_path)
            return None
        if extension in ["wav", "mp3"]:
            return load_audio_values(safe_input_path, training)
        if extension in ["csv", "xlsx", "json"]:
            return load_table_values(safe_input_path, extension)
        if extension == "txt":
            return load_text_values(safe_input_path)
        if extension in iio_formats:
            return load_image_values(safe_input_path)
        return load_video_values(safe_input_path)
    except Exception as error:
        catch(error)
        return None


def load_remote_dataset(
    src: str,
    revision: str | None,
    sample_rows: int | None = None,
):
    from datasets import load_dataset

    split_name = "train"
    if sample_rows is not None and sample_rows > 0:
        split_name = f"train[:{int(sample_rows)}]"
    if revision:
        return load_dataset(src, revision=revision, split=split_name)
    return load_dataset(src, split=split_name)


def load_remote_dataset_fallback(
    src: str,
    url_type: str,
    revision: str | None,
    sample_rows: int | None = None,
):
    from datasets import load_dataset

    split_name = "train"
    if sample_rows is not None and sample_rows > 0:
        split_name = f"train[:{int(sample_rows)}]"
    if revision:
        return load_dataset(
            url_type,
            data_files={"train": src},
            revision=revision,
            split=split_name,
        )
    return load_dataset(
        url_type,
        data_files={"train": src},
        split=split_name,
    )


def load_dataset_attempt(
    load_dataset_call: Callable[[], object],
    source_name: str,
) -> tuple[object | None, bool]:
    try:
        return load_dataset_call(), True
    except FileNotFoundError:
        import logging

        logging.error(f"Dataset {source_name} not found.")
        return None, False
    except ConnectionError:
        import logging

        logging.error(f"Connection error while loading dataset {source_name}.")
        return None, False
    except Exception as error:
        import logging

        logging.error(f"Error loading dataset {source_name}: {error}")
        return None, True


def fetch_dataset(
    src: str,
    url_type: str | None = None,
    revision: str | None = None,
    sample_rows: int | None = None,
):
    dataset, allow_fallback = _load_dataset_attempt(
        lambda: _load_remote_dataset(src, revision, sample_rows=sample_rows),
        src,
    )
    if dataset is not None or url_type is None or not allow_fallback:
        return dataset
    fallback_source = f"{url_type} with data_files {src}"
    fallback_dataset, _ = _load_dataset_attempt(
        lambda: _load_remote_dataset_fallback(
            src,
            url_type,
            revision,
            sample_rows=sample_rows,
        ),
        fallback_source,
    )
    return fallback_dataset


def load_source(
    remote_src: str | None = None,
    features: Iterable[str] | None = None,
    labels: Iterable[str] | None = None,
    url_type: str | None = None,
    revision: str | None = None,
):
    active_runtime = _runtime()
    if remote_src:
        fetch = getattr(active_runtime, "fetch_dataset", fetch_dataset)
        return fetch(remote_src, url_type, revision)
    if features:
        build_dataset = getattr(
            active_runtime,
            "files_to_dataset",
            files_to_dataset,
        )
        return build_dataset(features, labels)
    return None


def is_string_array(value, numpy_module) -> bool:
    return isinstance(value, numpy_module.ndarray) and (
        numpy_module.issubdtype(value.dtype, numpy_module.str_)
        or numpy_module.issubdtype(value.dtype, numpy_module.object_)
    )


def loaded_values_have_strings(loaded, numpy_module) -> bool:
    if is_string_array(loaded, numpy_module):
        return True
    if isinstance(loaded, list):
        return any(
            is_string_array(item, numpy_module)
            for item in loaded
            if item is not None
        )
    return False


def append_loaded_values(target: list, loaded, convert_value) -> None:
    if isinstance(loaded, list):
        target.extend(
            convert_value(item) for item in loaded if item is not None
        )
        return
    target.append(convert_value(loaded))


def collect_loaded_values(
    paths: Iterable[str], role: str, numpy_module
) -> tuple[list, bool] | None:
    active_runtime = _runtime()
    values = []
    has_strings = False
    for path in paths:
        loaded = load_as_numpy(path, training=True)
        if loaded is None:
            if role == "feature":
                active_runtime.logger.exception(
                    f"Error loading feature file: {path}"
                )
            else:
                _catch(Exception(f"Error loading label file: {path}"))
            return None
        has_strings = has_strings or loaded_values_have_strings(
            loaded, numpy_module
        )
        append_loaded_values(values, loaded, active_runtime.cupy_to_numpy)
    return values, has_strings


def stringify_loaded_values(values, numpy_module) -> list[str]:
    stringified = []
    for value in values:
        if isinstance(value, numpy_module.ndarray):
            stringified.append(" ".join(value.astype(str).flatten()))
        else:
            stringified.append(str(value))
    return stringified


def tokenize_loaded_values(values, numpy_module):
    active_runtime = _runtime()
    tokenized_values = active_runtime.tokenize_and_pad(
        stringify_loaded_values(values, numpy_module),
        active_runtime.init_tokenizer(),
    )
    return [active_runtime.cupy_to_numpy(row) for row in tokenized_values]


def stack_tensor_rows(values, max_lengths, active_runtime):
    try:
        import torch
    except Exception:
        return np.stack(
            [
                active_runtime.cupy_to_numpy(
                    active_runtime.reshape_numpy(value, lengths=max_lengths)
                )
                for value in values
            ],
            axis=0,
        )
    return active_runtime.convert_tensor_dtype(
        torch.stack(
            [
                torch.tensor(
                    active_runtime.reshape_numpy(value, lengths=max_lengths)
                )
                for value in values
            ]
        )
    )


def build_tensor_dataset(features, labels, active_runtime):
    from definers.data.lightweight_datasets import resolve_tensor_dataset

    all_data = list(features) + list(labels)
    if not all_data:
        return None
    tensor_dataset_cls = resolve_tensor_dataset()
    max_lengths = active_runtime.get_max_shapes(*all_data)
    features_tensor = stack_tensor_rows(features, max_lengths, active_runtime)
    if labels:
        labels_tensor = stack_tensor_rows(labels, max_lengths, active_runtime)
        return tensor_dataset_cls(features_tensor, labels_tensor)
    return tensor_dataset_cls(features_tensor)


def files_to_dataset(features_paths, labels_paths=None):
    active_runtime = _runtime()
    try:
        feature_batch = collect_loaded_values(features_paths, "feature", np)
        if feature_batch is None:
            return None
        features, features_have_strings = feature_batch
        labels = []
        labels_have_strings = False
        if labels_paths:
            label_batch = collect_loaded_values(labels_paths, "label", np)
            if label_batch is None:
                return None
            labels, labels_have_strings = label_batch
    except Exception as error:
        _catch(error)
        return None
    if not features and not labels:
        active_runtime.logger.warning("No valid data loaded.")
        return None
    if features_have_strings:
        features = tokenize_loaded_values(features, np)
    if labels_paths and labels_have_strings:
        labels = tokenize_loaded_values(labels, np)
    try:
        return build_tensor_dataset(features, labels, active_runtime)
    except Exception as tensor_error:
        _catch(
            Exception(f"Error creating tensor dataset: {type(tensor_error)}")
        )
        _catch(tensor_error)
        return None


_catch = catch
_load_dataset_attempt = load_dataset_attempt
_load_remote_dataset = load_remote_dataset
_load_remote_dataset_fallback = load_remote_dataset_fallback
_runtime = runtime
