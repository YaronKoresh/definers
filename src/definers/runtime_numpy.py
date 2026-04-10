import contextlib
import importlib.util
import threading
import types

_PATCH_LOCK = threading.RLock()
_PATCHED_FINFO_ATTR = "_definers_patched_finfo"
_ORIGINAL_FINFO_ATTR = "_definers_original_finfo"
_RUNTIME_ARRAY_MODULE = None
_RUNTIME_NUMPY_MODULE = None


class _NoNep50Warning(contextlib.ContextDecorator):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        _args = [exc_type, exc, tb]
        return False


def find_spec(mod_name: str):
    try:
        return importlib.util.find_spec(mod_name)
    except Exception:
        return None


def set_aliases(module, aliases) -> None:
    if module is None:
        return
    namespace = getattr(module, "__dict__", None)
    for alias, target in aliases.items():
        if namespace is not None and alias in namespace:
            continue
        if namespace is None and hasattr(module, alias):
            continue
        setattr(module, alias, target)


def ensure_no_nep50_warning(numpy_module):
    if hasattr(numpy_module, "_no_nep50_warning"):
        return numpy_module._no_nep50_warning

    def dummy_npwarn_decorator_factory():
        return _NoNep50Warning()

    numpy_module._no_nep50_warning = dummy_npwarn_decorator_factory
    return numpy_module._no_nep50_warning


def _patch_array_module_aliases(np_module, numpy_module) -> None:
    set_aliases(
        np_module,
        {
            "float": getattr(np_module, "float64", numpy_module.float64),
            "int": getattr(np_module, "int64", numpy_module.int64),
            "bool": getattr(np_module, "bool_", numpy_module.bool_),
            "complex": getattr(
                np_module,
                "complex128",
                numpy_module.complex128,
            ),
            "object": getattr(np_module, "object_", numpy_module.object_),
            "str": getattr(np_module, "str_", numpy_module.str_),
            "unicode": getattr(np_module, "str_", numpy_module.str_),
            "inf": getattr(np_module, "inf", float("inf")),
            "Inf": getattr(np_module, "inf", float("inf")),
        },
    )
    for alias, target_name in (
        ("round_", "round"),
        ("product", "prod"),
        ("cumproduct", "cumprod"),
        ("alltrue", "all"),
        ("sometrue", "any"),
        ("rank", "ndim"),
    ):
        target = getattr(np_module, target_name, None)
        if target is not None:
            set_aliases(np_module, {alias: target})


def _patch_char_namespace(numpy_module) -> None:
    if "char" not in getattr(numpy_module, "__dict__", {}):
        numpy_module.char = types.SimpleNamespace()
    char_funcs = {
        "encode": lambda s, encoding=None: bytes(s, encoding or "utf-8"),
        "decode": lambda b, encoding=None: b.decode(encoding or "utf-8"),
        "lower": lambda s: s.lower(),
        "upper": lambda s: s.upper(),
        "capitalize": lambda s: s.capitalize(),
        "casefold": lambda s: s.casefold(),
        "title": lambda s: s.title(),
        "swapcase": lambda s: s.swapcase(),
        "startswith": lambda s, prefix, *args: s.startswith(prefix, *args),
        "endswith": lambda s, suffix, *args: s.endswith(suffix, *args),
        "strip": lambda s, chars=None: (
            s.strip(chars) if chars is not None else s.strip()
        ),
        "lstrip": lambda s, chars=None: (
            s.lstrip(chars) if chars is not None else s.lstrip()
        ),
        "rstrip": lambda s, chars=None: (
            s.rstrip(chars) if chars is not None else s.rstrip()
        ),
        "replace": lambda s, old, new, count=-1: (
            s.replace(old, new, count) if count != -1 else s.replace(old, new)
        ),
        "split": lambda s, sep=None, maxsplit=-1: (
            s.split(sep, maxsplit) if sep is not None else s.split()
        ),
        "rsplit": lambda s, sep=None, maxsplit=-1: (
            s.rsplit(sep, maxsplit) if sep is not None else s.rsplit()
        ),
        "splitlines": lambda s, keepends=False: s.splitlines(keepends),
        "partition": lambda s, sep: s.partition(sep),
        "rpartition": lambda s, sep: s.rpartition(sep),
        "join": lambda sep, iterable: sep.join(iterable),
        "count": lambda s, sub, start=None, end=None: (
            s.count(sub, start, end)
            if start is not None and end is not None
            else s.count(sub, start)
            if start is not None
            else s.count(sub)
        ),
        "find": lambda s, sub, start=None, end=None: (
            s.find(sub, start, end)
            if start is not None and end is not None
            else s.find(sub, start)
            if start is not None
            else s.find(sub)
        ),
        "rfind": lambda s, sub, start=None, end=None: (
            s.rfind(sub, start, end)
            if start is not None and end is not None
            else s.rfind(sub, start)
            if start is not None
            else s.rfind(sub)
        ),
        "index": lambda s, sub, start=None, end=None: (
            s.index(sub, start, end)
            if start is not None and end is not None
            else s.index(sub, start)
            if start is not None
            else s.index(sub)
        ),
        "rindex": lambda s, sub, start=None, end=None: (
            s.rindex(sub, start, end)
            if start is not None and end is not None
            else s.rindex(sub, start)
            if start is not None
            else s.rindex(sub)
        ),
        "zfill": lambda s, width: s.zfill(width),
        "center": lambda s, width, fillchar=" ": s.center(width, fillchar),
        "ljust": lambda s, width, fillchar=" ": s.ljust(width, fillchar),
        "rjust": lambda s, width, fillchar=" ": s.rjust(width, fillchar),
        "isalpha": lambda s: s.isalpha(),
        "isalnum": lambda s: s.isalnum(),
        "isdigit": lambda s: s.isdigit(),
        "isdecimal": lambda s: s.isdecimal(),
        "isnumeric": lambda s: s.isnumeric(),
        "isspace": lambda s: s.isspace(),
        "islower": lambda s: s.islower(),
        "isupper": lambda s: s.isupper(),
        "istitle": lambda s: s.istitle(),
        "add": lambda a, b: a + b,
        "multiply": lambda a, n: a * n,
        "mod": lambda s, values: s % values,
        "string_": lambda s: str(s),
        "bytes_": lambda s: (
            bytes(s, "utf-8") if not isinstance(s, bytes) else s
        ),
        "equal": lambda a, b: a == b,
        "not_equal": lambda a, b: a != b,
        "greater": lambda a, b: a > b,
        "greater_equal": lambda a, b: a >= b,
        "less": lambda a, b: a < b,
        "less_equal": lambda a, b: a <= b,
    }
    for name, func in char_funcs.items():
        if not hasattr(numpy_module.char, name):
            setattr(numpy_module.char, name, func)


def _patch_rec_namespace(numpy_module, recfunctions) -> None:
    rec_namespace = getattr(numpy_module, "rec", None)
    if rec_namespace is None:
        rec_namespace = types.SimpleNamespace()
    set_aliases(
        rec_namespace,
        {
            "append_fields": lambda base, names, data, dtypes=None: (
                recfunctions.append_fields(base, names, data, dtypes=dtypes)
            ),
            "drop_fields": lambda base, names: recfunctions.drop_fields(
                base,
                names,
            ),
            "rename_fields": lambda base, name_dict: recfunctions.rename_fields(
                base,
                name_dict,
            ),
            "merge_arrays": lambda arrays, fill_value=-1, flatten=False: (
                recfunctions.merge_arrays(
                    arrays,
                    fill_value=fill_value,
                    flatten=flatten,
                )
            ),
        },
    )
    if hasattr(numpy_module, "recarray") and not hasattr(
        rec_namespace,
        "recarray",
    ):
        rec_namespace.recarray = numpy_module.recarray
    numpy_module.rec = rec_namespace


def _patch_machar(numpy_module) -> None:
    if hasattr(numpy_module, "MachAr") and hasattr(numpy_module, "machar"):
        return

    class MachAr:
        def __init__(self, dtype=None):
            dtype = dtype or numpy_module.float64
            info = numpy_module.finfo(dtype)
            self.dtype = dtype
            self.bits = info.bits
            self.eps = info.eps
            self.epsneg = getattr(info, "epsneg", None)
            self.machep = getattr(info, "machep", None)
            self.negep = getattr(info, "negep", None)
            self.iexp = getattr(info, "iexp", None)
            self.maxexp = getattr(info, "maxexp", None)
            self.minexp = getattr(info, "minexp", None)
            self.max = info.max
            self.min = info.min
            self.tiny = info.tiny
            self.resolution = getattr(info, "resolution", None)

        @classmethod
        def from_dtype(cls, dtype):
            return cls(dtype)

        def get_finfo(self):
            return numpy_module.finfo(self.dtype)

        def __repr__(self):
            return f"<MachAr dtype={self.dtype}>"

        @staticmethod
        def apply_patches(target, patches, overwrite=False):
            for name, value in patches.items():
                if overwrite or not hasattr(target, name):
                    setattr(target, name, value)
            return target

    machar_namespace = types.SimpleNamespace(MachAr=MachAr)
    core_module = getattr(numpy_module, "core", None)
    if core_module is not None and not hasattr(core_module, "machar"):
        core_module.machar = machar_namespace
    if not hasattr(numpy_module, "MachAr"):
        numpy_module.MachAr = MachAr
    if not hasattr(numpy_module, "machar"):
        numpy_module.machar = machar_namespace


def _patch_testing_namespace(numpy_module) -> None:
    if hasattr(numpy_module, "testing") and not hasattr(
        numpy_module.testing,
        "Tester",
    ):

        class Tester:
            def test(self, label="fast", _extra_argv=None):
                return True

        numpy_module.testing.Tester = Tester


def _patch_distutils_namespace(numpy_module) -> None:
    distutils_namespace = getattr(numpy_module, "distutils", None)
    if distutils_namespace is None:
        distutils_namespace = types.SimpleNamespace()
    misc_util = getattr(distutils_namespace, "misc_util", None)
    if misc_util is None:
        misc_util = types.SimpleNamespace()
    if not hasattr(misc_util, "get_info"):
        misc_util.get_info = lambda *args, **kwargs: {}
    distutils_namespace.misc_util = misc_util
    system_info = getattr(distutils_namespace, "system_info", None)
    if system_info is None:
        system_info = types.SimpleNamespace()
    if not hasattr(system_info, "get_info"):
        system_info.get_info = misc_util.get_info
    distutils_namespace.system_info = system_info
    numpy_module.distutils = distutils_namespace


def _patch_finfo(numpy_module) -> None:
    current_finfo = numpy_module.finfo
    if getattr(current_finfo, _PATCHED_FINFO_ATTR, False):
        return
    original_finfo = getattr(numpy_module, _ORIGINAL_FINFO_ATTR, current_finfo)

    def patched_finfo(dtype):
        try:
            return original_finfo(dtype)
        except (TypeError, ValueError):
            resolved_dtype = numpy_module.dtype(dtype)
            if numpy_module.issubdtype(resolved_dtype, numpy_module.integer):
                return numpy_module.iinfo(resolved_dtype)
            raise

    setattr(patched_finfo, _PATCHED_FINFO_ATTR, True)
    setattr(numpy_module, _ORIGINAL_FINFO_ATTR, original_finfo)
    numpy_module.finfo = patched_finfo


def ensure_numpy_compatibility(numpy_module):
    from numpy.lib import recfunctions

    with _PATCH_LOCK:
        _patch_array_module_aliases(numpy_module, numpy_module)
        set_aliases(
            numpy_module,
            {
                "intp": numpy_module.int_,
                "string_": numpy_module.bytes_,
                "strings": numpy_module.bytes_,
            },
        )
        _patch_char_namespace(numpy_module)
        if "asscalar" not in getattr(numpy_module, "__dict__", {}):
            numpy_module.asscalar = lambda a: a.item()
        _patch_rec_namespace(numpy_module, recfunctions)
        _patch_machar(numpy_module)
        _patch_testing_namespace(numpy_module)
        _patch_distutils_namespace(numpy_module)
        if "set_string_function" not in getattr(numpy_module, "__dict__", {}):
            numpy_module.set_string_function = lambda *args, **kwargs: None
        _patch_finfo(numpy_module)
        ensure_no_nep50_warning(numpy_module)
    return numpy_module


def _load_cupy_module():
    if find_spec("cupy") is None:
        return None
    try:
        import cupy as cupy_module

        return cupy_module
    except Exception:
        return None


def _is_cupy_runtime_available(cupy_module) -> bool:
    availability_probe = getattr(cupy_module, "is_available", None)
    if callable(availability_probe):
        try:
            return bool(availability_probe())
        except Exception:
            pass
    runtime_module = getattr(
        getattr(cupy_module, "cuda", None), "runtime", None
    )
    get_device_count = getattr(runtime_module, "getDeviceCount", None)
    if not callable(get_device_count):
        return False
    try:
        return int(get_device_count()) > 0
    except Exception:
        return False


def bootstrap_runtime_numpy(force: bool = False):
    global _RUNTIME_ARRAY_MODULE, _RUNTIME_NUMPY_MODULE

    with _PATCH_LOCK:
        if (
            not force
            and _RUNTIME_ARRAY_MODULE is not None
            and _RUNTIME_NUMPY_MODULE is not None
        ):
            return _RUNTIME_ARRAY_MODULE, _RUNTIME_NUMPY_MODULE

        import numpy as numpy_module

        ensure_numpy_compatibility(numpy_module)
        np_module = numpy_module
        cupy_module = _load_cupy_module()
        if cupy_module is not None and _is_cupy_runtime_available(cupy_module):
            _patch_array_module_aliases(cupy_module, numpy_module)
            np_module = cupy_module
        _RUNTIME_ARRAY_MODULE = np_module
        _RUNTIME_NUMPY_MODULE = numpy_module
        return np_module, numpy_module


def patch_numpy_runtime(force: bool = False):
    return bootstrap_runtime_numpy(force=force)


def init_cupy_numpy():
    return bootstrap_runtime_numpy()


def get_array_module():
    np_module, _ = bootstrap_runtime_numpy()
    return np_module


def get_numpy_module():
    _, numpy_module = bootstrap_runtime_numpy()
    return numpy_module


def runtime_backend_name(force: bool = False) -> str:
    np_module, numpy_module = bootstrap_runtime_numpy(force=force)
    if np_module is numpy_module:
        return "numpy"
    return getattr(np_module, "__name__", "cupy")


def is_cupy_backend(force: bool = False) -> bool:
    return runtime_backend_name(force=force).split(".", 1)[0] == "cupy"


def runtime_backend_info(force: bool = False) -> dict[str, object]:
    np_module, numpy_module = bootstrap_runtime_numpy(force=force)
    backend_name = runtime_backend_name(force=force)
    return {
        "array_module": backend_name,
        "numpy_module": getattr(numpy_module, "__name__", "numpy"),
        "numpy_version": getattr(numpy_module, "__version__", None),
        "cupy_enabled": np_module is not numpy_module,
        "cupy_version": getattr(np_module, "__version__", None)
        if np_module is not numpy_module
        else None,
    }


__all__ = (
    "bootstrap_runtime_numpy",
    "ensure_no_nep50_warning",
    "ensure_numpy_compatibility",
    "find_spec",
    "get_array_module",
    "get_numpy_module",
    "init_cupy_numpy",
    "is_cupy_backend",
    "patch_numpy_runtime",
    "runtime_backend_info",
    "runtime_backend_name",
    "set_aliases",
)
