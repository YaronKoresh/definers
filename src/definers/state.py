from __future__ import annotations

from collections.abc import Iterable, Iterator, MutableMapping
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

DEFAULT_RUNTIME_SCOPE = "default"


def _normalize_scope(scope: str) -> str:
    normalized_scope = str(scope).strip()
    if not normalized_scope:
        raise ValueError("scope must not be empty")
    return normalized_scope


def _reset_mapping(
    target: MutableMapping[str, Any], values: dict[str, Any]
) -> None:
    target.clear()
    target.update(values)


class LockedMapping(MutableMapping[str, Any]):
    __slots__ = ("_data", "_lock")

    def __init__(
        self,
        initial: MutableMapping[str, Any] | dict[str, Any] | None = None,
        *,
        lock: RLock | None = None,
    ) -> None:
        self._data: dict[str, Any] = {}
        self._lock = lock or RLock()
        if initial:
            self.update(initial)

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._data[key]

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(tuple(self._data))

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def update(self, other: Any = (), /, **kwargs: Any) -> None:
        with self._lock:
            if isinstance(other, MutableMapping):
                for key, value in other.items():
                    self._data[str(key)] = value
            else:
                for key, value in dict(other).items():
                    self._data[str(key)] = value
            for key, value in kwargs.items():
                self._data[str(key)] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.setdefault(key, default)

    def pop(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if default is None and key not in self._data:
                raise KeyError(key)
            return self._data.pop(key, default)


def _wrap_mapping(
    values: MutableMapping[str, Any] | dict[str, Any],
    lock: RLock,
) -> LockedMapping:
    if isinstance(values, LockedMapping):
        return values
    return LockedMapping(values, lock=lock)


def _wrap_tokenizers(
    values: MutableMapping[str, dict[str, Any]] | dict[str, dict[str, Any]],
    lock: RLock,
) -> LockedMapping:
    wrapped_values = {
        key: _wrap_mapping(value, lock) for key, value in dict(values).items()
    }
    return LockedMapping(wrapped_values, lock=lock)


def _default_models() -> dict[str, Any]:
    return {
        "video": None,
        "image": None,
        "upscale": None,
        "detect": None,
        "answer": None,
        "summary": None,
        "translate": None,
        "music": None,
        "song": None,
        "speech-recognition": None,
        "audio-classification": None,
        "tts": None,
        "stable-whisper": None,
    }


def _default_tokenizers() -> dict[str, dict[str, Any]]:
    return {
        "summary": {"tokenizer": None, "model_name": None},
        "translate": {"tokenizer": None, "model_name": None},
        "general": {"tokenizer": None, "model_name": None},
    }


def _default_processors() -> dict[str, Any]:
    return {"answer": None, "music": None}


def _default_configs() -> dict[str, Any]:
    return {"answer": None}


@dataclass(frozen=True, slots=True)
class RuntimeCollections:
    models: MutableMapping[str, Any]
    tokenizers: MutableMapping[str, dict[str, Any]]
    processors: MutableMapping[str, Any]
    configs: MutableMapping[str, Any]


@dataclass(slots=True)
class RuntimeState:
    models: dict[str, Any] = field(default_factory=_default_models)
    tokenizers: dict[str, dict[str, Any]] = field(
        default_factory=_default_tokenizers
    )
    processors: dict[str, Any] = field(default_factory=_default_processors)
    configs: dict[str, Any] = field(default_factory=_default_configs)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.models = _wrap_mapping(self.models, self._lock)
        self.tokenizers = _wrap_tokenizers(self.tokenizers, self._lock)
        self.processors = _wrap_mapping(self.processors, self._lock)
        self.configs = _wrap_mapping(self.configs, self._lock)

    def _get_tokenizer_entry_mapping(
        self, name: str
    ) -> MutableMapping[str, Any]:
        with self._lock:
            entry = self.tokenizers.get(name)
            if entry is None:
                entry = LockedMapping(lock=self._lock)
                self.tokenizers[name] = entry
            elif not isinstance(entry, LockedMapping):
                entry = _wrap_mapping(entry, self._lock)
                self.tokenizers[name] = entry
            return entry

    def get_collections(self) -> RuntimeCollections:
        return RuntimeCollections(
            models=self.models,
            tokenizers=self.tokenizers,
            processors=self.processors,
            configs=self.configs,
        )

    def get_model(self, name: str, default: Any = None) -> Any:
        return self.models.get(name, default)

    def set_model(self, name: str, value: Any) -> Any:
        self.models[name] = value
        return value

    def get_tokenizer_entry(
        self, name: str, default: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        return self.tokenizers.get(name, default)

    def get_tokenizer(self, name: str, default: Any = None) -> Any:
        entry = self.tokenizers.get(name)
        if entry is None:
            return default
        return entry.get("tokenizer", default)

    def set_tokenizer(
        self,
        name: str,
        tokenizer: Any,
        *,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        entry = self._get_tokenizer_entry_mapping(name)
        entry["tokenizer"] = tokenizer
        if model_name is not None:
            entry["model_name"] = model_name
        elif "model_name" not in entry:
            entry["model_name"] = None
        return entry

    def get_processor(self, name: str, default: Any = None) -> Any:
        return self.processors.get(name, default)

    def set_processor(self, name: str, value: Any) -> Any:
        self.processors[name] = value
        return value

    def get_config(self, name: str, default: Any = None) -> Any:
        return self.configs.get(name, default)

    def set_config(self, name: str, value: Any) -> Any:
        self.configs[name] = value
        return value

    def reset(self) -> None:
        with self._lock:
            _reset_mapping(self.models, _default_models())
            _reset_mapping(
                self.tokenizers,
                _wrap_tokenizers(_default_tokenizers(), self._lock),
            )
            _reset_mapping(self.processors, _default_processors())
            _reset_mapping(self.configs, _default_configs())


@dataclass(slots=True)
class RuntimeStateRegistry:
    states: dict[str, RuntimeState] = field(default_factory=dict)
    default_scope: str = DEFAULT_RUNTIME_SCOPE
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        with self._lock:
            self.default_scope = _normalize_scope(self.default_scope)
            if self.default_scope not in self.states:
                self.states[self.default_scope] = RuntimeState()

    def get_state(self, scope: str = DEFAULT_RUNTIME_SCOPE) -> RuntimeState:
        normalized_scope = _normalize_scope(scope)
        with self._lock:
            state = self.states.get(normalized_scope)
            if state is None:
                state = RuntimeState()
                self.states[normalized_scope] = state
            return state

    def create_state(
        self,
        scope: str,
        *,
        replace: bool = False,
    ) -> RuntimeState:
        normalized_scope = _normalize_scope(scope)
        with self._lock:
            if replace and normalized_scope == self.default_scope:
                state = self.states.setdefault(normalized_scope, RuntimeState())
                state.reset()
                return state
            if not replace and normalized_scope in self.states:
                return self.states[normalized_scope]
            state = RuntimeState()
            self.states[normalized_scope] = state
            return state

    def delete_state(self, scope: str) -> None:
        normalized_scope = _normalize_scope(scope)
        if normalized_scope == self.default_scope:
            raise ValueError(
                f"{self.default_scope} runtime scope cannot be deleted"
            )
        with self._lock:
            self.states.pop(normalized_scope, None)

    def list_scopes(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self.states))

    def reset(self, scope: str = DEFAULT_RUNTIME_SCOPE) -> RuntimeState:
        state = self.get_state(scope)
        state.reset()
        return state

    def reset_many(self, scopes: Iterable[str]) -> None:
        for scope in scopes:
            self.reset(scope)


RUNTIME_STATES = RuntimeStateRegistry()
RUNTIME_STATE = RUNTIME_STATES.get_state(DEFAULT_RUNTIME_SCOPE)
DEFAULT_RUNTIME_COLLECTIONS = RUNTIME_STATE.get_collections()
MODELS = DEFAULT_RUNTIME_COLLECTIONS.models
TOKENIZERS = DEFAULT_RUNTIME_COLLECTIONS.tokenizers
PROCESSORS = DEFAULT_RUNTIME_COLLECTIONS.processors
CONFIGS = DEFAULT_RUNTIME_COLLECTIONS.configs


def get_runtime_state(scope: str = DEFAULT_RUNTIME_SCOPE) -> RuntimeState:
    return RUNTIME_STATES.get_state(scope)


def get_default_runtime_state() -> RuntimeState:
    return get_runtime_state(DEFAULT_RUNTIME_SCOPE)


def get_runtime_collections(
    scope: str = DEFAULT_RUNTIME_SCOPE,
) -> RuntimeCollections:
    return get_runtime_state(scope).get_collections()


def get_default_runtime_collections() -> RuntimeCollections:
    return get_runtime_collections(DEFAULT_RUNTIME_SCOPE)


def get_runtime_models(
    scope: str = DEFAULT_RUNTIME_SCOPE,
) -> MutableMapping[str, Any]:
    return get_runtime_collections(scope).models


def get_runtime_tokenizers(
    scope: str = DEFAULT_RUNTIME_SCOPE,
) -> MutableMapping[str, dict[str, Any]]:
    return get_runtime_collections(scope).tokenizers


def get_runtime_processors(
    scope: str = DEFAULT_RUNTIME_SCOPE,
) -> MutableMapping[str, Any]:
    return get_runtime_collections(scope).processors


def get_runtime_configs(
    scope: str = DEFAULT_RUNTIME_SCOPE,
) -> MutableMapping[str, Any]:
    return get_runtime_collections(scope).configs


def create_runtime_state(
    scope: str,
    *,
    replace: bool = False,
) -> RuntimeState:
    return RUNTIME_STATES.create_state(scope, replace=replace)


def delete_runtime_state(scope: str) -> None:
    RUNTIME_STATES.delete_state(scope)


def list_runtime_scopes() -> tuple[str, ...]:
    return RUNTIME_STATES.list_scopes()


def get_model(name: str, default: Any = None) -> Any:
    return RUNTIME_STATE.get_model(name, default)


def set_model(name: str, value: Any) -> Any:
    return RUNTIME_STATE.set_model(name, value)


def get_tokenizer(name: str, default: Any = None) -> Any:
    return RUNTIME_STATE.get_tokenizer(name, default)


def get_tokenizer_entry(
    name: str, default: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    return RUNTIME_STATE.get_tokenizer_entry(name, default)


def set_tokenizer(
    name: str,
    tokenizer: Any,
    *,
    model_name: str | None = None,
) -> dict[str, Any]:
    return RUNTIME_STATE.set_tokenizer(name, tokenizer, model_name=model_name)


def get_processor(name: str, default: Any = None) -> Any:
    return RUNTIME_STATE.get_processor(name, default)


def set_processor(name: str, value: Any) -> Any:
    return RUNTIME_STATE.set_processor(name, value)


def get_config(name: str, default: Any = None) -> Any:
    return RUNTIME_STATE.get_config(name, default)


def set_config(name: str, value: Any) -> Any:
    return RUNTIME_STATE.set_config(name, value)


def reset_runtime_state(scope: str = "default") -> None:
    RUNTIME_STATES.reset(scope)


__all__ = (
    "CONFIGS",
    "DEFAULT_RUNTIME_COLLECTIONS",
    "DEFAULT_RUNTIME_SCOPE",
    "LockedMapping",
    "MODELS",
    "PROCESSORS",
    "RUNTIME_STATE",
    "RUNTIME_STATES",
    "RuntimeCollections",
    "RuntimeState",
    "RuntimeStateRegistry",
    "TOKENIZERS",
    "create_runtime_state",
    "delete_runtime_state",
    "get_config",
    "get_default_runtime_collections",
    "get_default_runtime_state",
    "get_model",
    "get_processor",
    "get_runtime_collections",
    "get_runtime_configs",
    "get_runtime_models",
    "get_runtime_processors",
    "get_runtime_state",
    "get_runtime_tokenizers",
    "get_tokenizer",
    "get_tokenizer_entry",
    "list_runtime_scopes",
    "reset_runtime_state",
    "set_config",
    "set_model",
    "set_processor",
    "set_tokenizer",
)
