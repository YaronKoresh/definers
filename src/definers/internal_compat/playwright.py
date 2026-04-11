from __future__ import annotations

import sys
from types import ModuleType
from typing import Any


def _missing_playwright(*_args: Any, **_kwargs: Any):
    raise ImportError("playwright is not available")


class _MissingPlaywrightContextManager:
    def __enter__(self):
        raise ImportError("playwright is not available")

    def __exit__(self, _exc_type, _exc, _tb):
        return False


def sync_playwright():
    return _MissingPlaywrightContextManager()


def expect(*args: Any, **kwargs: Any):
    return _missing_playwright(*args, **kwargs)


sync_api = ModuleType("playwright.sync_api")
sync_api.expect = expect
sync_api.sync_playwright = sync_playwright

playwright = ModuleType("playwright")
playwright.sync_api = sync_api

sys.modules.setdefault("playwright", playwright)
sys.modules.setdefault("playwright.sync_api", sync_api)

__all__ = ("expect", "playwright", "sync_api", "sync_playwright")
