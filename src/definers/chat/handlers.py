from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from definers.chat.audit_message_builder import (
    build_chat_audit_message,
)
from definers.chat.contracts import (
    Answerer,
    ChatAuditMessageBuilder,
    ChatHistory,
    ChatMediaAttachment,
    ChatMessage,
    ChatRequest,
    ChatRequestContext,
    ChatRequestMetadata,
    ChatRequestNormalizer,
    ChatTextContext,
    HistoryItem,
    LanguageDetector,
    Logger,
    Translator,
    Validator,
)

MEDIA_INSTRUCTION = "and please read the media from my new message carefully"


def create_chat_request(
    message: ChatMessage,
    history: ChatHistory | None = None,
) -> ChatRequest:
    copied_message = {str(key): value for key, value in message.items()}
    copied_history = (
        () if history is None else tuple(dict(item) for item in history)
    )
    return ChatRequest(
        message=copied_message,
        history=copied_history,
    )


def coerce_chat_request(
    request: ChatRequest | ChatMessage,
    history: ChatHistory | None,
) -> ChatRequest:
    if isinstance(request, ChatRequest):
        return create_chat_request(request.message, request.history)
    return create_chat_request(request, history)


def _copy_history(history: ChatHistory) -> list[HistoryItem]:
    return [dict(item) for item in history]


def _normalize_files(message: ChatMessage) -> tuple[str, ...]:
    files = message.get("files", [])
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
        return ()
    normalized_files: list[str] = []
    for file_path in files:
        normalized_file_path = str(file_path).strip()
        if normalized_file_path:
            normalized_files.append(normalized_file_path)
    return tuple(normalized_files)


def _normalize_media(
    message: ChatMessage,
) -> tuple[ChatMediaAttachment, ...]:
    return tuple(
        ChatMediaAttachment(path=file_path)
        for file_path in _normalize_files(message)
    )


def _normalize_text(
    message: ChatMessage,
    *,
    detect_language: LanguageDetector,
    translate_text: Translator,
    validate_text: Validator,
) -> ChatTextContext | None:
    raw_text = message.get("text")
    if raw_text is None:
        return None
    text_value = str(raw_text).strip()
    if not text_value:
        return None
    original_language = detect_language(text_value)
    translated_text = text_value
    if original_language != "en":
        translated_text = translate_text(text_value)
    validated_text = str(validate_text(translated_text)).strip()
    if not validated_text:
        return None
    return ChatTextContext(
        raw=text_value,
        original_language=original_language,
        translated=translated_text,
        validated=validated_text,
    )


def _build_history(
    base_history: list[HistoryItem],
    media: tuple[ChatMediaAttachment, ...],
    text_context: ChatTextContext | None,
) -> tuple[list[HistoryItem], tuple[str, ...]]:
    normalized_history = [dict(item) for item in base_history]
    included_types: list[str] = []
    if media:
        included_types.append("files")
        normalized_history.extend(
            {"role": "user", "content": {"path": item.path}} for item in media
        )
    if text_context is not None:
        included_types.append("text")
        normalized_history.append(
            {"role": "user", "content": text_context.validated}
        )
        if media:
            normalized_history.append(
                {"role": "user", "content": MEDIA_INSTRUCTION}
            )
    return normalized_history, tuple(included_types)


def normalize_chat_request(
    request: ChatRequest | ChatMessage,
    history: ChatHistory | None = None,
    *,
    detect_language: LanguageDetector,
    translate_text: Translator,
    validate_text: Validator,
) -> ChatRequestContext:
    normalized_request = coerce_chat_request(request, history)
    base_history = _copy_history(normalized_request.history)
    media = _normalize_media(normalized_request.message)
    text_context = _normalize_text(
        normalized_request.message,
        detect_language=detect_language,
        translate_text=translate_text,
        validate_text=validate_text,
    )
    normalized_history, included_types = _build_history(
        base_history,
        media,
        text_context,
    )
    return ChatRequestContext(
        request=normalized_request,
        metadata=ChatRequestMetadata(
            includes_text=text_context is not None,
            includes_media=bool(media),
        ),
        base_history=base_history,
        history=normalized_history,
        included_types=included_types,
        original_language=(
            None if text_context is None else text_context.original_language
        ),
        media=media,
        text=text_context,
    )


def log_chat_request(log: Logger, build_audit_message, context) -> None:
    log("Chat", build_audit_message(context))


def log_chat_response(log: Logger, response) -> None:
    log("Chatbot response", response)


@dataclass(frozen=True, slots=True)
class ChatRequestHandler:
    normalize_request: ChatRequestNormalizer
    build_audit_message: ChatAuditMessageBuilder
    answer: Answerer
    log: Logger

    def handle(
        self,
        request: ChatRequest | ChatMessage,
        history: ChatHistory | None = None,
    ) -> Any:
        normalized_request = coerce_chat_request(request, history)
        context = self.normalize_request(normalized_request)
        log_chat_request(self.log, self.build_audit_message, context)
        response = self.answer(context.history)
        log_chat_response(self.log, response)
        return response


def create_chat_request_handler(
    *,
    detect_language: LanguageDetector,
    translate_text: Translator,
    validate_text: Validator,
    answer: Answerer,
    log: Logger,
    language_names: Mapping[str, str],
):
    return ChatRequestHandler(
        normalize_request=lambda request: normalize_chat_request(
            request,
            detect_language=detect_language,
            translate_text=translate_text,
            validate_text=validate_text,
        ),
        build_audit_message=lambda context: build_chat_audit_message(
            context.included_types,
            context.original_language,
            language_names,
        ),
        answer=answer,
        log=log,
    )


def handle_chat_request(
    request: ChatMessage,
    history: ChatHistory | None = None,
    *,
    detect_language: LanguageDetector,
    translate_text: Translator,
    validate_text: Validator,
    answer: Answerer,
    log: Logger,
    language_names: Mapping[str, str],
) -> Any:
    return create_chat_request_handler(
        detect_language=detect_language,
        translate_text=translate_text,
        validate_text=validate_text,
        answer=answer,
        log=log,
        language_names=language_names,
    ).handle(request, history)
