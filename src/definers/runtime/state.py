from collections.abc import Iterable, MutableMapping
from dataclasses import dataclass, field
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
        entry = self.tokenizers.setdefault(name, {})
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
        _reset_mapping(self.models, _default_models())
        _reset_mapping(self.tokenizers, _default_tokenizers())
        _reset_mapping(self.processors, _default_processors())
        _reset_mapping(self.configs, _default_configs())


@dataclass(slots=True)
class RuntimeStateRegistry:
    states: dict[str, RuntimeState] = field(default_factory=dict)
    default_scope: str = DEFAULT_RUNTIME_SCOPE

    def __post_init__(self) -> None:
        self.default_scope = _normalize_scope(self.default_scope)
        if self.default_scope not in self.states:
            self.states[self.default_scope] = RuntimeState()

    def get_state(self, scope: str = DEFAULT_RUNTIME_SCOPE) -> RuntimeState:
        return self.states.setdefault(_normalize_scope(scope), RuntimeState())

    def create_state(
        self,
        scope: str,
        *,
        replace: bool = False,
    ) -> RuntimeState:
        normalized_scope = _normalize_scope(scope)
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
        self.states.pop(normalized_scope, None)

    def list_scopes(self) -> tuple[str, ...]:
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
