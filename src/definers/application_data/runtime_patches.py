class RuntimePatchService:
    @staticmethod
    def find_spec(mod_name: str):
        import importlib

        try:
            module = importlib.import_module(mod_name)
            return module.__spec__
        except Exception:
            return None

    @staticmethod
    def set_aliases(module, aliases) -> None:
        for alias, target in aliases.items():
            if alias not in getattr(module, "__dict__", {}):
                setattr(module, alias, target)

    @classmethod
    def init_cupy_numpy(cls):
        import sys
        import types

        import numpy as numpy_module
        from numpy.lib import recfunctions

        np_module = None
        cupy_in_sys = sys.modules.get("cupy")
        if cupy_in_sys is not None and cls.find_spec("cupy"):
            np_module = cupy_in_sys
        if np_module is None:
            np_module = numpy_module
        if "float" not in getattr(np_module, "__dict__", {}):
            np_module.float = np_module.float64
        if "int" not in getattr(np_module, "__dict__", {}):
            np_module.int = np_module.int64
        cls.set_aliases(
            numpy_module,
            {
                "intp": numpy_module.int_,
                "float": numpy_module.float64,
                "int": numpy_module.int64,
                "bool": numpy_module.bool_,
                "complex": numpy_module.complex128,
                "object": numpy_module.object_,
                "str": numpy_module.str_,
                "str_": numpy_module.str_,
                "string_": numpy_module.bytes_,
                "strings": numpy_module.bytes_,
                "unicode": numpy_module.str_,
                "inf": float("inf"),
                "Inf": float("inf"),
            },
        )
        cls.set_aliases(
            numpy_module,
            {
                "round_": numpy_module.round,
                "product": numpy_module.prod,
                "cumproduct": numpy_module.cumprod,
                "alltrue": numpy_module.all,
                "sometrue": numpy_module.any,
                "rank": numpy_module.ndim,
            },
        )
        if "char" not in getattr(numpy_module, "__dict__", {}):
            setattr(numpy_module, "char", types.SimpleNamespace())
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
                s.replace(old, new, count)
                if count != -1
                else s.replace(old, new)
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
        if "asscalar" not in getattr(numpy_module, "__dict__", {}):
            numpy_module.asscalar = lambda a: a.item()
        if "rec" not in getattr(numpy_module, "__dict__", {}):

            class NumpyRec:
                @staticmethod
                def append_fields(base, names, data, dtypes=None):
                    return recfunctions.append_fields(
                        base, names, data, dtypes=dtypes
                    )

                @staticmethod
                def drop_fields(base, names):
                    return recfunctions.drop_fields(base, names)

                @staticmethod
                def rename_fields(base, name_dict):
                    return recfunctions.rename_fields(base, name_dict)

                @staticmethod
                def merge_arrays(arrays, fill_value=-1, flatten=False):
                    return recfunctions.merge_arrays(
                        arrays, fill_value=fill_value, flatten=flatten
                    )

            numpy_module.rec = NumpyRec()
        if "machar" not in getattr(numpy_module, "__dict__", {}):

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

            numpy_module.core.machar = MachAr()
        if hasattr(numpy_module, "testing") and not hasattr(
            numpy_module.testing, "Tester"
        ):

            class Tester:
                def test(self, label="fast", _extra_argv=None):
                    return True

            numpy_module.testing.Tester = Tester
        if "distutils" not in getattr(numpy_module, "__dict__", {}):

            class DummyDistutils:
                class MiscUtils:
                    def get_info(self, *args, **kwargs):
                        return {}

            numpy_module.distutils = DummyDistutils()
        if "set_string_function" not in getattr(numpy_module, "__dict__", {}):
            numpy_module.set_string_function = lambda *args, **kwargs: None
        original_finfo = numpy_module.finfo

        def patched_finfo(dtype):
            try:
                return original_finfo(dtype)
            except TypeError:
                return numpy_module.iinfo(dtype)

        numpy_module.finfo = patched_finfo
        if "_no_nep50_warning" not in getattr(numpy_module, "__dict__", {}):

            def dummy_npwarn_decorator_factory():
                def npwarn_decorator(value):
                    return value

                return npwarn_decorator

            numpy_module._no_nep50_warning = dummy_npwarn_decorator_factory
        return np_module, numpy_module


find_spec = RuntimePatchService.find_spec
init_cupy_numpy = RuntimePatchService.init_cupy_numpy
