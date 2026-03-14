from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from definers.application_chat.contracts import (
    Answerer,
    ChatAuditMessageBuilder,
    ChatHistory,
    ChatMediaAttachment,
    ChatMessage,
    ChatRequest,
    ChatRequestContext,
    ChatRequestNormalizer,
    ChatTextContext,
    HistoryItem,
    LanguageDetector,
    Logger,
    Translator,
    Validator,
)


MEDIA_INSTRUCTION = "and please read the media from my new message carefully"


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
        normalized_request = _coerce_chat_request(request, history)
        context = self.normalize_request(normalized_request)
        self.log("Chat", self.build_audit_message(context))
        response = self.answer(context.history)
        self.log("Chatbot response", response)
        return response


def _coerce_chat_request(
    request: ChatRequest | ChatMessage,
    history: ChatHistory | None,
) -> ChatRequest:
    if isinstance(request, ChatRequest):
        return request
    return create_chat_request(request, history)


def create_chat_request(
    message: ChatMessage,
    history: ChatHistory | None = None,
) -> ChatRequest:
    return ChatRequest(message=message, history=() if history is None else history)


def _copy_history(history: ChatHistory) -> list[HistoryItem]:
    return [dict(item) for item in history]


def _normalize_files(message: ChatMessage) -> tuple[str, ...]:
    files = message.get("files", [])
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
        return ()
    return tuple(str(file_path) for file_path in files if str(file_path))


def _normalize_media(message: ChatMessage) -> tuple[ChatMediaAttachment, ...]:
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
    if not raw_text:
        return None
    text_value = str(raw_text)
    original_language = detect_language(text_value)
    translated_text = text_value
    if original_language != "en":
        translated_text = translate_text(text_value)
    validated_text = validate_text(translated_text)
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
            {"role": "user", "content": {"path": item.path}}
            for item in media
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
    normalized_request = _coerce_chat_request(request, history)
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
        base_history=base_history,
        history=normalized_history,
        included_types=included_types,
        original_language=(
            None if text_context is None else text_context.original_language
        ),
        media=media,
        text=text_context,
    )


def build_chat_audit_message(
    included_types: Sequence[str],
    original_language: str | None,
    language_names: Mapping[str, str],
) -> str:
    line_break = "\n"
    included = line_break.join(included_types)
    if original_language is None:
        return (
            f"Got a new message.{line_break}{line_break}"
            f"The message including the following types of data:{line_break}{included}"
        )
    language_name = language_names.get(original_language, original_language)
    return (
        f"Got a new message in {language_name}.{line_break}{line_break}"
        f"The message including the following types of data:{line_break}{included}"
    )


def create_chat_request_handler(
    *,
    detect_language: LanguageDetector,
    translate_text: Translator,
    validate_text: Validator,
    answer: Answerer,
    log: Logger,
    language_names: Mapping[str, str],
) -> ChatRequestHandler:
    def normalize_request(request: ChatRequest) -> ChatRequestContext:
        return normalize_chat_request(
            request,
            detect_language=detect_language,
            translate_text=translate_text,
            validate_text=validate_text,
        )

    def build_audit_message(context: ChatRequestContext) -> str:
        return build_chat_audit_message(
            context.included_types,
            context.original_language,
            language_names,
        )

    return ChatRequestHandler(
        normalize_request=normalize_request,
        build_audit_message=build_audit_message,
        answer=answer,
        log=log,
    )


def handle_chat_request(
    request: ChatRequest | ChatMessage,
    history: ChatHistory | None = None,
    *,
    detect_language: LanguageDetector,
    translate_text: Translator,
    validate_text: Validator,
    answer: Answerer,
    log: Logger,
    language_names: Mapping[str, str],
) -> Any:
    handler = create_chat_request_handler(
        detect_language=detect_language,
        translate_text=translate_text,
        validate_text=validate_text,
        answer=answer,
        log=log,
        language_names=language_names,
    )
    return handler.handle(request, history)
