import pickle

import joblib.numpy_pickle

_BLOCKED_GLOBALS = frozenset(
    {
        ("builtins", "__import__"),
        ("builtins", "compile"),
        ("builtins", "eval"),
        ("builtins", "exec"),
        ("builtins", "input"),
        ("importlib", "import_module"),
        ("marshal", "loads"),
        ("nt", "system"),
        ("os", "system"),
        ("pickle", "load"),
        ("pickle", "loads"),
        ("posix", "system"),
        ("runpy", "run_module"),
        ("runpy", "run_path"),
        ("subprocess", "call"),
        ("subprocess", "check_call"),
        ("subprocess", "check_output"),
        ("subprocess", "getoutput"),
        ("subprocess", "getstatusoutput"),
        ("subprocess", "popen"),
        ("subprocess", "run"),
    }
)
_HTML_PREFIXES = (b"<!doctype html", b"<html")
_LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_SUPPORTED_TYPES = frozenset({"joblib", "pkl"})


def _normalize_model_type(model_type: str) -> str:
    normalized_model_type = str(model_type).strip().lower().lstrip(".")
    if normalized_model_type not in _SUPPORTED_TYPES:
        raise ValueError(f"Unsupported serialized model type: '{model_type}'.")
    return normalized_model_type


def _reject_blocked_global(module: str, name: str) -> None:
    normalized_global = (str(module).strip().lower(), str(name).strip().lower())
    if normalized_global in _BLOCKED_GLOBALS:
        raise ValueError(
            "Unsafe serialized model rejected: "
            f"{module}.{name} is blocked during deserialization."
        )


class _GuardedUnpicklerMixin:
    def find_class(self, module: str, name: str):
        _reject_blocked_global(module, name)
        return super().find_class(module, name)


class GuardedPickleUnpickler(_GuardedUnpicklerMixin, pickle.Unpickler):
    pass


class GuardedJoblibUnpickler(
    _GuardedUnpicklerMixin, joblib.numpy_pickle.NumpyUnpickler
):
    pass


def validate_serialized_model_file(path: str, model_type: str) -> None:
    _normalize_model_type(model_type)
    with open(path, "rb") as file_obj:
        header = file_obj.read(512)
    lowered_header = header.lstrip().lower()
    if lowered_header.startswith(_LFS_PREFIX):
        raise ValueError(
            "Downloaded a Git LFS pointer instead of serialized model bytes."
        )
    if lowered_header.startswith(_HTML_PREFIXES):
        raise ValueError("Downloaded HTML instead of serialized model bytes.")


def load_serialized_model(path: str, model_type: str):
    normalized_model_type = _normalize_model_type(model_type)
    validate_serialized_model_file(path, normalized_model_type)
    if normalized_model_type == "joblib":
        return _load_joblib_model(path)
    return _load_pickle_model(path)


def _load_pickle_model(path: str):
    with open(path, "rb") as file_obj:
        return GuardedPickleUnpickler(file_obj).load()


def _load_joblib_model(path: str):
    with open(path, "rb") as file_obj:
        with joblib.numpy_pickle._validate_fileobject_and_memmap(
            file_obj,
            path,
            None,
        ) as (validated_file_obj, validated_mmap_mode):
            if isinstance(validated_file_obj, str):
                raise ValueError(
                    "Legacy joblib persistence format is not supported for secure model loading."
                )
            try:
                return GuardedJoblibUnpickler(
                    path,
                    validated_file_obj,
                    True,
                    mmap_mode=validated_mmap_mode,
                ).load()
            except UnicodeDecodeError as error:
                value_error = ValueError(
                    "Python 2 joblib payloads are not supported for secure model loading."
                )
                value_error.__cause__ = error
                raise value_error
