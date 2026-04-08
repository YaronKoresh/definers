class DatasetTensorBuilder:
    @staticmethod
    def is_string_array(value, numpy_module) -> bool:
        return isinstance(value, numpy_module.ndarray) and (
            numpy_module.issubdtype(value.dtype, numpy_module.str_)
            or numpy_module.issubdtype(value.dtype, numpy_module.object_)
        )

    @classmethod
    def loaded_values_have_strings(cls, loaded, numpy_module) -> bool:
        if cls.is_string_array(loaded, numpy_module):
            return True
        if isinstance(loaded, list):
            return any(
                cls.is_string_array(item, numpy_module)
                for item in loaded
                if item is not None
            )
        return False

    @staticmethod
    def append_loaded_values(target, loaded, convert_value) -> None:
        if isinstance(loaded, list):
            target.extend(
                convert_value(item) for item in loaded if item is not None
            )
            return
        target.append(convert_value(loaded))

    @classmethod
    def collect_loaded_values(cls, paths, role: str, numpy_module):
        import definers.data.loaders as loaders_module

        runtime = loaders_module._runtime()
        values = []
        has_strings = False
        for path in paths:
            loaded = loaders_module.load_as_numpy(path, training=True)
            if loaded is None:
                if role == "feature":
                    runtime.logger.exception(
                        f"Error loading feature file: {path}"
                    )
                else:
                    loaders_module._catch(
                        Exception(f"Error loading label file: {path}")
                    )
                return None
            has_strings = has_strings or cls.loaded_values_have_strings(
                loaded, numpy_module
            )
            cls.append_loaded_values(values, loaded, runtime.cupy_to_numpy)
        return values, has_strings

    @staticmethod
    def stringify_loaded_values(values, numpy_module) -> list[str]:
        stringified = []
        for value in values:
            if isinstance(value, numpy_module.ndarray):
                stringified.append(" ".join(value.astype(str).flatten()))
            else:
                stringified.append(str(value))
        return stringified

    @staticmethod
    def tokenize_loaded_values(values, numpy_module):
        import definers.data.loaders as loaders_module

        runtime = loaders_module._runtime()
        tokenized_values = runtime.tokenize_and_pad(
            loaders_module._stringify_loaded_values(values, numpy_module),
            runtime.init_tokenizer(),
        )
        return [runtime.cupy_to_numpy(row) for row in tokenized_values]

    @staticmethod
    def stack_tensor_rows(values, max_lengths, runtime):
        try:
            import torch
        except Exception:
            import numpy as np

            return np.stack(
                [
                    runtime.cupy_to_numpy(
                        runtime.reshape_numpy(value, lengths=max_lengths)
                    )
                    for value in values
                ],
                axis=0,
            )
        return runtime.convert_tensor_dtype(
            torch.stack(
                [
                    torch.tensor(
                        runtime.reshape_numpy(value, lengths=max_lengths)
                    )
                    for value in values
                ]
            )
        )

    @staticmethod
    def build_tensor_dataset(features, labels, runtime):
        from definers.data.lightweight_datasets import (
            resolve_tensor_dataset,
        )

        all_data = list(features) + list(labels)
        if not all_data:
            return None
        tensor_dataset_cls = resolve_tensor_dataset()
        max_lengths = runtime.get_max_shapes(*all_data)
        features_tensor = DatasetTensorBuilder.stack_tensor_rows(
            features, max_lengths, runtime
        )
        if labels:
            labels_tensor = DatasetTensorBuilder.stack_tensor_rows(
                labels, max_lengths, runtime
            )
            return tensor_dataset_cls(features_tensor, labels_tensor)
        return tensor_dataset_cls(features_tensor)

    @classmethod
    def files_to_dataset(cls, features_paths, labels_paths=None):
        import numpy as np

        import definers.data.loaders as loaders_module

        runtime = loaders_module._runtime()
        try:
            feature_batch = loaders_module._collect_loaded_values(
                features_paths, "feature", np
            )
            if feature_batch is None:
                return None
            features, features_have_strings = feature_batch
            labels = []
            labels_have_strings = False
            if labels_paths:
                label_batch = loaders_module._collect_loaded_values(
                    labels_paths, "label", np
                )
                if label_batch is None:
                    return None
                labels, labels_have_strings = label_batch
        except Exception as error:
            loaders_module._catch(error)
            return None
        if not features and not labels:
            runtime.logger.warning("No valid data loaded.")
            return None
        if features_have_strings:
            features = loaders_module._tokenize_loaded_values(features, np)
        if labels_paths and labels_have_strings:
            labels = loaders_module._tokenize_loaded_values(labels, np)
        try:
            return loaders_module._build_tensor_dataset(
                features, labels, runtime
            )
        except Exception as tensor_error:
            loaders_module._catch(
                Exception(
                    f"Error creating tensor dataset: {type(tensor_error)}"
                )
            )
            loaders_module._catch(tensor_error)
            return None
