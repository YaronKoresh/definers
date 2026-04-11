from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

ChatMessage = Mapping[str, object]
ChatHistory = Sequence[ChatMessage]
HistoryItem = dict[str, object]


@dataclass(frozen=True, slots=True)
class ChatMediaAttachment:
    path: str


@dataclass(frozen=True, slots=True)
class ChatTextContext:
    raw: str
    original_language: str
    translated: str
    validated: str


@dataclass(frozen=True, slots=True)
class ChatRequest:
    message: ChatMessage
    history: ChatHistory


@dataclass(frozen=True, slots=True)
class ChatRequestMetadata:
    includes_text: bool
    includes_media: bool


@dataclass(frozen=True, slots=True)
class ChatRequestContext:
    request: ChatRequest
    metadata: ChatRequestMetadata
    base_history: list[HistoryItem]
    history: list[HistoryItem]
    included_types: tuple[str, ...]
    original_language: str | None
    media: tuple[ChatMediaAttachment, ...]
    text: ChatTextContext | None


class Answerer(Protocol): ...


class Logger(Protocol): ...


class LanguageDetector(Protocol): ...


class Translator(Protocol): ...


class Validator(Protocol): ...


class ChatRequestNormalizer(Protocol): ...


class ChatAuditMessageBuilder(Protocol): ...
