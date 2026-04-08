from __future__ import annotations

import importlib


def librosa_module():
    return importlib.import_module("librosa")


def njit(*args, **kwargs):
    try:
        from numba import njit as numba_njit

        return numba_njit(*args, **kwargs)
    except Exception:
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(function):
            return function

        return decorator
