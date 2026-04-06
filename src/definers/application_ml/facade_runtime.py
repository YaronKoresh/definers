from __future__ import annotations

from types import SimpleNamespace


class MlFacadeRuntime:
    @staticmethod
    def build_answer_runtime(models, processors):
        return SimpleNamespace(MODELS=models, PROCESSORS=processors)

    @staticmethod
    def build_linear_regression_runtime(device_fn):
        return SimpleNamespace(device=device_fn)

    @staticmethod
    def build_train_linear_regression_runtime(
        device_fn,
        initialize_linear_regression_fn,
    ):
        return SimpleNamespace(
            device=device_fn,
            initialize_linear_regression=initialize_linear_regression_fn,
        )

    @staticmethod
    def build_training_array_adapter(
        catch_fn,
        cupy_to_numpy_fn,
        get_max_shapes_fn,
        numpy_to_cupy_fn,
        reshape_numpy_fn,
    ):
        return SimpleNamespace(
            catch=catch_fn,
            cupy_to_numpy=cupy_to_numpy_fn,
            get_max_shapes=get_max_shapes_fn,
            numpy_to_cupy=numpy_to_cupy_fn,
            reshape_numpy=reshape_numpy_fn,
        )

    @staticmethod
    def resolve_training_row_concatenate(np_module):
        cupy_module = getattr(np_module, "cuda", None)
        if cupy_module is not None:
            concatenate = getattr(cupy_module, "cupy", None)
            if concatenate is not None:
                concatenate = getattr(concatenate, "concatenate", None)
                if concatenate is not None:
                    return concatenate
        return np_module.concatenate
