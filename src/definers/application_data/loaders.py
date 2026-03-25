class LoaderService:
    @staticmethod
    def catch(error: Exception) -> None:
        from definers.application_data.loader_runtime import (
            LoaderRuntimeSupport,
        )

        return LoaderRuntimeSupport.catch(error)

    @staticmethod
    def delete(path: str | None) -> None:
        from definers.application_data.loader_runtime import (
            LoaderRuntimeSupport,
        )

        return LoaderRuntimeSupport.delete(path)

    @staticmethod
    def read(path: str):
        from definers.application_data.loader_runtime import (
            LoaderRuntimeSupport,
        )

        return LoaderRuntimeSupport.read(path)

    @staticmethod
    def tmp(extension: str) -> str:
        from definers.application_data.loader_runtime import (
            LoaderRuntimeSupport,
        )

        return LoaderRuntimeSupport.tmp(extension)

    @staticmethod
    def runtime():
        from definers.application_data.loader_runtime import (
            LoaderRuntimeSupport,
        )

        return LoaderRuntimeSupport.runtime()

    @staticmethod
    def has_parameter(values) -> bool:
        from definers.application_data.dataset_shape_service import (
            DatasetShapeService,
        )

        return DatasetShapeService.has_parameter(values)

    @staticmethod
    def column_names(dataset) -> tuple[str, ...]:
        from definers.application_data.dataset_shape_service import (
            DatasetShapeService,
        )

        return DatasetShapeService.column_names(dataset)

    @classmethod
    def selected_columns(cls, dataset, requested_columns) -> list[str]:
        from definers.application_data.dataset_shape_service import (
            DatasetShapeService,
        )

        return DatasetShapeService.selected_columns(dataset, requested_columns)

    @classmethod
    def dataset_slice(
        cls, dataset, start_index: int, end_index: int
    ) -> dict[str, object]:
        from definers.application_data.dataset_shape_service import (
            DatasetShapeService,
        )

        return DatasetShapeService.dataset_slice(
            dataset, start_index, end_index
        )

    @staticmethod
    def batched_column_values(data, label_names):
        from definers.application_data.dataset_shape_service import (
            DatasetShapeService,
        )

        return DatasetShapeService.batched_column_values(data, label_names)

    @staticmethod
    def path_extension(path: str) -> str | None:
        from definers.application_data.dataset_shape_service import (
            DatasetShapeService,
        )

        return DatasetShapeService.path_extension(path)

    @staticmethod
    def load_audio_values(path: str, training: bool):
        from definers.application_data.dataset_value_loader import (
            DatasetValueLoader,
        )

        return DatasetValueLoader.load_audio_values(path, training)

    @staticmethod
    def load_table_values(path: str, extension: str):
        from definers.application_data.dataset_value_loader import (
            DatasetValueLoader,
        )

        return DatasetValueLoader.load_table_values(path, extension)

    @staticmethod
    def load_text_values(path: str):
        from definers.application_data.dataset_value_loader import (
            DatasetValueLoader,
        )

        return DatasetValueLoader.load_text_values(path)

    @staticmethod
    def load_image_values(path: str):
        from definers.application_data.dataset_value_loader import (
            DatasetValueLoader,
        )

        return DatasetValueLoader.load_image_values(path)

    @staticmethod
    def load_video_values(path: str):
        from definers.application_data.dataset_value_loader import (
            DatasetValueLoader,
        )

        return DatasetValueLoader.load_video_values(path)

    @classmethod
    def load_as_numpy(cls, path: str, training: bool = False):
        from definers.application_data.dataset_value_loader import (
            DatasetValueLoader,
        )

        return DatasetValueLoader.load_as_numpy(path, training)

    @staticmethod
    def load_remote_dataset(src: str, revision: str | None):
        from definers.application_data.dataset_source_loader import (
            DatasetSourceLoader,
        )

        return DatasetSourceLoader.load_remote_dataset(src, revision)

    @staticmethod
    def load_remote_dataset_fallback(
        src: str, url_type: str, revision: str | None
    ):
        from definers.application_data.dataset_source_loader import (
            DatasetSourceLoader,
        )

        return DatasetSourceLoader.load_remote_dataset_fallback(
            src, url_type, revision
        )

    @staticmethod
    def load_dataset_attempt(load_dataset_call, source_name: str):
        from definers.application_data.dataset_source_loader import (
            DatasetSourceLoader,
        )

        return DatasetSourceLoader.load_dataset_attempt(
            load_dataset_call, source_name
        )

    @staticmethod
    def fetch_dataset(
        src: str, url_type: str | None = None, revision: str | None = None
    ):
        from definers.application_data.dataset_source_loader import (
            DatasetSourceLoader,
        )

        return DatasetSourceLoader.fetch_dataset(src, url_type, revision)

    @classmethod
    def drop_columns(cls, dataset, drop_list):
        from definers.application_data.dataset_shape_service import (
            DatasetShapeService,
        )

        return DatasetShapeService.drop_columns(dataset, drop_list)

    @classmethod
    def select_columns(cls, dataset, cols):
        from definers.application_data.dataset_shape_service import (
            DatasetShapeService,
        )

        return DatasetShapeService.select_columns(dataset, cols)

    @classmethod
    def select_rows(cls, dataset, start_index: int, end_index: int):
        from definers.application_data.dataset_shape_service import (
            DatasetShapeService,
        )

        return DatasetShapeService.select_rows(dataset, start_index, end_index)

    @staticmethod
    def split_columns(data, labels, is_batch: bool = False):
        from definers.application_data.dataset_shape_service import (
            DatasetShapeService,
        )

        return DatasetShapeService.split_columns(data, labels, is_batch)

    @staticmethod
    def is_string_array(value, numpy_module) -> bool:
        from definers.application_data.dataset_tensor_builder import (
            DatasetTensorBuilder,
        )

        return DatasetTensorBuilder.is_string_array(value, numpy_module)

    @staticmethod
    def loaded_values_have_strings(loaded, numpy_module) -> bool:
        from definers.application_data.dataset_tensor_builder import (
            DatasetTensorBuilder,
        )

        return DatasetTensorBuilder.loaded_values_have_strings(
            loaded, numpy_module
        )

    @staticmethod
    def append_loaded_values(target, loaded, convert_value) -> None:
        from definers.application_data.dataset_tensor_builder import (
            DatasetTensorBuilder,
        )

        return DatasetTensorBuilder.append_loaded_values(
            target, loaded, convert_value
        )

    @staticmethod
    def collect_loaded_values(paths, role: str, numpy_module):
        from definers.application_data.dataset_tensor_builder import (
            DatasetTensorBuilder,
        )

        return DatasetTensorBuilder.collect_loaded_values(
            paths, role, numpy_module
        )

    @staticmethod
    def stringify_loaded_values(values, numpy_module) -> list[str]:
        from definers.application_data.dataset_tensor_builder import (
            DatasetTensorBuilder,
        )

        return DatasetTensorBuilder.stringify_loaded_values(
            values, numpy_module
        )

    @staticmethod
    def tokenize_loaded_values(values, numpy_module):
        from definers.application_data.dataset_tensor_builder import (
            DatasetTensorBuilder,
        )

        return DatasetTensorBuilder.tokenize_loaded_values(values, numpy_module)

    @staticmethod
    def stack_tensor_rows(values, max_lengths, runtime):
        from definers.application_data.dataset_tensor_builder import (
            DatasetTensorBuilder,
        )

        return DatasetTensorBuilder.stack_tensor_rows(
            values, max_lengths, runtime
        )

    @staticmethod
    def build_tensor_dataset(features, labels, runtime):
        from definers.application_data.dataset_tensor_builder import (
            DatasetTensorBuilder,
        )

        return DatasetTensorBuilder.build_tensor_dataset(
            features, labels, runtime
        )

    @staticmethod
    def files_to_dataset(features_paths, labels_paths=None):
        from definers.application_data.dataset_tensor_builder import (
            DatasetTensorBuilder,
        )

        return DatasetTensorBuilder.files_to_dataset(
            features_paths, labels_paths
        )

    @staticmethod
    def load_source(
        remote_src: str | None = None,
        features=None,
        labels=None,
        url_type: str | None = None,
        revision: str | None = None,
    ):
        from definers.application_data.dataset_source_loader import (
            DatasetSourceLoader,
        )

        return DatasetSourceLoader.load_source(
            remote_src=remote_src,
            features=features,
            labels=labels,
            url_type=url_type,
            revision=revision,
        )


_catch = LoaderService.catch
_delete = LoaderService.delete
_read = LoaderService.read
_tmp = LoaderService.tmp
_runtime = LoaderService.runtime
_has_parameter = LoaderService.has_parameter
_column_names = LoaderService.column_names
_selected_columns = LoaderService.selected_columns
_dataset_slice = LoaderService.dataset_slice
_batched_column_values = LoaderService.batched_column_values
_path_extension = LoaderService.path_extension
_load_audio_values = LoaderService.load_audio_values
_load_table_values = LoaderService.load_table_values
_load_text_values = LoaderService.load_text_values
_load_image_values = LoaderService.load_image_values
_load_video_values = LoaderService.load_video_values
load_as_numpy = LoaderService.load_as_numpy
_load_remote_dataset = LoaderService.load_remote_dataset
_load_remote_dataset_fallback = LoaderService.load_remote_dataset_fallback
_load_dataset_attempt = LoaderService.load_dataset_attempt
fetch_dataset = LoaderService.fetch_dataset
drop_columns = LoaderService.drop_columns
select_columns = LoaderService.select_columns
select_rows = LoaderService.select_rows
split_columns = LoaderService.split_columns
_is_string_array = LoaderService.is_string_array
_loaded_values_have_strings = LoaderService.loaded_values_have_strings
_append_loaded_values = LoaderService.append_loaded_values
_collect_loaded_values = LoaderService.collect_loaded_values
_stringify_loaded_values = LoaderService.stringify_loaded_values
_tokenize_loaded_values = LoaderService.tokenize_loaded_values
_stack_tensor_rows = LoaderService.stack_tensor_rows
_build_tensor_dataset = LoaderService.build_tensor_dataset
files_to_dataset = LoaderService.files_to_dataset
load_source = LoaderService.load_source
