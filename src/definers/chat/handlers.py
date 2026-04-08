from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from definers.chat.audit_message_builder import (
    ChatAuditMessageBuilderService,
)
from definers.chat.contracts import (
    Answerer,
    ChatAuditMessageBuilder,
    ChatHistory,
    ChatMessage,
    ChatRequest,
    ChatRequestContext,
    ChatRequestNormalizer,
    Logger,
)
from definers.chat.handler_factory import ChatRequestHandlerFactory
from definers.chat.history_builder import ChatHistoryBuilder
from definers.chat.request_audit_logger import (
    ChatRequestAuditLogger,
)
from definers.chat.request_factory import ChatRequestFactory
from definers.chat.request_normalizer import (
    ChatRequestNormalizerService,
)


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
        normalized_request = ChatRequestFactory.coerce_chat_request(
            request,
            history,
        )
        context = self.normalize_request(normalized_request)
        ChatRequestAuditLogger.log_request(
            self.log,
            self.build_audit_message,
            context,
        )
        response = self.answer(context.history)
        ChatRequestAuditLogger.log_response(self.log, response)
        return response


MEDIA_INSTRUCTION = ChatHistoryBuilder.media_instruction
create_chat_request = ChatRequestFactory.create_chat_request
normalize_chat_request = ChatRequestNormalizerService.normalize_chat_request
build_chat_audit_message = (
    ChatAuditMessageBuilderService.build_chat_audit_message
)
create_chat_request_handler = (
    ChatRequestHandlerFactory.create_chat_request_handler
)
handle_chat_request = ChatRequestHandlerFactory.handle_chat_request
