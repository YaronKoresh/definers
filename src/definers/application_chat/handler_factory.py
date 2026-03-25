from collections.abc import Mapping
from typing import Any

from definers.application_chat.audit_message_builder import (
    ChatAuditMessageBuilderService,
)
from definers.application_chat.contracts import (
    Answerer,
    ChatHistory,
    ChatMessage,
    LanguageDetector,
    Logger,
    Translator,
    Validator,
)
from definers.application_chat.request_normalizer import ChatRequestNormalizerService


class ChatRequestHandlerFactory:
    @classmethod
    def create_chat_request_handler(
        cls,
        *,
        detect_language: LanguageDetector,
        translate_text: Translator,
        validate_text: Validator,
        answer: Answerer,
        log: Logger,
        language_names: Mapping[str, str],
    ):
        from definers.application_chat.handlers import ChatRequestHandler

        return ChatRequestHandler(
            normalize_request=lambda request: ChatRequestNormalizerService.normalize_chat_request(
                request,
                detect_language=detect_language,
                translate_text=translate_text,
                validate_text=validate_text,
            ),
            build_audit_message=lambda context: ChatAuditMessageBuilderService.build_chat_audit_message(
                context.included_types,
                context.original_language,
                language_names,
            ),
            answer=answer,
            log=log,
        )

    @classmethod
    def handle_chat_request(
        cls,
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
        handler = cls.create_chat_request_handler(
            detect_language=detect_language,
            translate_text=translate_text,
            validate_text=validate_text,
            answer=answer,
            log=log,
            language_names=language_names,
        )
        return handler.handle(request, history)