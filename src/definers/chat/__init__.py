from __future__ import annotations

import importlib
from typing import Any


def _module_exports(module_name: str, *names: str) -> dict[str, str]:
    return {name: module_name for name in names}


_LAZY_SUBMODULES = {
    "audit_message_builder",
    "contracts",
    "handler_factory",
    "handlers",
    "history_builder",
    "request_audit_logger",
    "request_context_assembler",
    "request_factory",
    "request_normalizer",
    "text_context_builder",
}

_CHAT_EXPORTS = {
    **_module_exports(
        "handlers",
        "MEDIA_INSTRUCTION",
        "build_chat_audit_message",
        "create_chat_request",
        "create_chat_request_handler",
        "handle_chat_request",
        "normalize_chat_request",
    ),
}

__all__ = (
    "MEDIA_INSTRUCTION",
    "audit_message_builder",
    "build_chat_audit_message",
    "contracts",
    "create_chat_request",
    "create_chat_request_handler",
    "handle_chat_request",
    "handler_factory",
    "handlers",
    "history_builder",
    "normalize_chat_request",
    "request_audit_logger",
    "request_context_assembler",
    "request_factory",
    "request_normalizer",
    "text_context_builder",
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    module_name = _CHAT_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(_LAZY_SUBMODULES).union(__all__))
