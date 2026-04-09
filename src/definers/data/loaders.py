def _loader_runtime_support():
    from definers.data.loader_runtime import LoaderRuntimeSupport

    return LoaderRuntimeSupport


def _shape_module():
    from definers.data.datasets import shape

    return shape


def _value_loader():
    from definers.data.datasets.value import DatasetValueLoader

    return DatasetValueLoader


def _source_loader():
    from definers.data.datasets.source import DatasetSourceLoader

    return DatasetSourceLoader


def _tensor_builder():
    from definers.data.datasets.tensor import DatasetTensorBuilder

    return DatasetTensorBuilder


def catch(error: Exception) -> None:
    return _loader_runtime_support().catch(error)


def delete(path: str | None) -> None:
    return _loader_runtime_support().delete(path)


def read(path: str):
    return _loader_runtime_support().read(path)


def tmp(extension: str) -> str:
    return _loader_runtime_support().tmp(extension)


def runtime():
    return _loader_runtime_support().runtime()


def has_parameter(values) -> bool:
    return _shape_module().has_parameter(values)


def column_names(dataset) -> tuple[str, ...]:
    return _shape_module().column_names(dataset)


def selected_columns(dataset, requested_columns) -> list[str]:
    return _shape_module().selected_columns(dataset, requested_columns)


def dataset_slice(
    dataset, start_index: int, end_index: int
) -> dict[str, object]:
    return _shape_module().dataset_slice(dataset, start_index, end_index)


def batched_column_values(data, label_names):
    return _shape_module().batched_column_values(data, label_names)


def path_extension(path: str) -> str | None:
    return _shape_module().path_extension(path)


def load_audio_values(path: str, training: bool):
    return _value_loader().load_audio_values(path, training)


def load_table_values(path: str, extension: str):
    return _value_loader().load_table_values(path, extension)


def load_text_values(path: str):
    return _value_loader().load_text_values(path)


def load_image_values(path: str):
    return _value_loader().load_image_values(path)


def load_video_values(path: str):
    return _value_loader().load_video_values(path)


def load_as_numpy(path: str, training: bool = False):
    return _value_loader().load_as_numpy(path, training)


def load_remote_dataset(src: str, revision: str | None):
    return _source_loader().load_remote_dataset(src, revision)


def load_remote_dataset_fallback(src: str, url_type: str, revision: str | None):
    return _source_loader().load_remote_dataset_fallback(
        src, url_type, revision
    )


def load_dataset_attempt(load_dataset_call, source_name: str):
    return _source_loader().load_dataset_attempt(load_dataset_call, source_name)


def fetch_dataset(
    src: str, url_type: str | None = None, revision: str | None = None
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


def drop_columns(dataset, drop_list):
    return _shape_module().drop_columns(dataset, drop_list)


def select_columns(dataset, cols):
    if not _has_parameter(cols):
        return dataset
    selected_names = set(cols)
    columns_to_drop = [
        column_name
        for column_name in _column_names(dataset)
        if column_name not in selected_names
    ]
    return drop_columns(dataset, columns_to_drop)


def select_rows(dataset, start_index: int, end_index: int):
    return _shape_module().select_rows(dataset, start_index, end_index)


def split_columns(data, labels, is_batch: bool = False):
    if not _has_parameter(labels):
        features, label_values = data
        return features, label_values
    if is_batch:
        return _batched_column_values(data, set(labels))
    features = drop_columns(data, labels)
    label_values = select_columns(data, labels)
    return features, label_values


def is_string_array(value, numpy_module) -> bool:
    return _tensor_builder().is_string_array(value, numpy_module)


def loaded_values_have_strings(loaded, numpy_module) -> bool:
    return _tensor_builder().loaded_values_have_strings(loaded, numpy_module)


def append_loaded_values(target, loaded, convert_value) -> None:
    return _tensor_builder().append_loaded_values(target, loaded, convert_value)


def collect_loaded_values(paths, role: str, numpy_module):
    return _tensor_builder().collect_loaded_values(paths, role, numpy_module)


def stringify_loaded_values(values, numpy_module) -> list[str]:
    return _tensor_builder().stringify_loaded_values(values, numpy_module)


def tokenize_loaded_values(values, numpy_module):
    return _tensor_builder().tokenize_loaded_values(values, numpy_module)


def stack_tensor_rows(values, max_lengths, runtime):
    return _tensor_builder().stack_tensor_rows(values, max_lengths, runtime)


def build_tensor_dataset(features, labels, runtime):
    return _tensor_builder().build_tensor_dataset(features, labels, runtime)


def files_to_dataset(features_paths, labels_paths=None):
    return _tensor_builder().files_to_dataset(features_paths, labels_paths)


def load_source(
    remote_src: str | None = None,
    features=None,
    labels=None,
    url_type: str | None = None,
    revision: str | None = None,
):
    active_runtime = _runtime()
    if remote_src:
        fetch = getattr(active_runtime, "fetch_dataset", fetch_dataset)
        return fetch(remote_src, url_type, revision)
    if features:
        build_dataset = getattr(
            active_runtime, "files_to_dataset", files_to_dataset
        )
        return build_dataset(features, labels)
    return None


_catch = catch
_delete = delete
_read = read
_tmp = tmp
_runtime = runtime
_has_parameter = has_parameter
_column_names = column_names
_selected_columns = selected_columns
_dataset_slice = dataset_slice
_batched_column_values = batched_column_values
_path_extension = path_extension
_load_audio_values = load_audio_values
_load_table_values = load_table_values
_load_text_values = load_text_values
_load_image_values = load_image_values
_load_video_values = load_video_values
_load_remote_dataset = load_remote_dataset
_load_remote_dataset_fallback = load_remote_dataset_fallback
_load_dataset_attempt = load_dataset_attempt
_is_string_array = is_string_array
_loaded_values_have_strings = loaded_values_have_strings
_append_loaded_values = append_loaded_values
_collect_loaded_values = collect_loaded_values
_stringify_loaded_values = stringify_loaded_values
_tokenize_loaded_values = tokenize_loaded_values
_stack_tensor_rows = stack_tensor_rows
_build_tensor_dataset = build_tensor_dataset
