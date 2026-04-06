from definers.application_chat.contracts import (
    ChatHistory,
    ChatMessage,
    ChatRequest,
    ChatRequestContext,
    LanguageDetector,
    Translator,
    Validator,
)
from definers.application_chat.request_context_assembler import (
    ChatRequestContextAssembler,
)
from definers.application_chat.request_factory import ChatRequestFactory


class ChatRequestNormalizerService:
    @classmethod
    def normalize_chat_request(
        cls,
        request: ChatRequest | ChatMessage,
        history: ChatHistory | None = None,
        *,
        detect_language: LanguageDetector,
        translate_text: Translator,
        validate_text: Validator,
    ) -> ChatRequestContext:
        normalized_request = ChatRequestFactory.coerce_chat_request(
            request, history
        )
        return ChatRequestContextAssembler.assemble_request_context(
            normalized_request,
            detect_language=detect_language,
            translate_text=translate_text,
            validate_text=validate_text,
        )
