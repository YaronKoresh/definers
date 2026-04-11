from collections.abc import Mapping
from typing import Any

from definers.chat.audit_message_builder import (
    build_chat_audit_message,
)
from definers.chat.contracts import (
    Answerer,
    ChatHistory,
    ChatMessage,
    LanguageDetector,
    Logger,
    Translator,
    Validator,
)
from definers.chat.request_normalizer import (
    normalize_chat_request,
)


def create_chat_request_handler(
    *,
    detect_language: LanguageDetector,
    translate_text: Translator,
    validate_text: Validator,
    answer: Answerer,
    log: Logger,
    language_names: Mapping[str, str],
):
    from definers.chat.handlers import ChatRequestHandler

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
    handler = create_chat_request_handler(
        detect_language=detect_language,
        translate_text=translate_text,
        validate_text=validate_text,
        answer=answer,
        log=log,
        language_names=language_names,
    )
    return handler.handle(request, history)
