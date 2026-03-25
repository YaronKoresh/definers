class ApplicationChatFacade:
    @staticmethod
    def get_contracts_module():
        import importlib

        return importlib.import_module("definers.application_chat.contracts")

    @staticmethod
    def get_handlers_module():
        import importlib

        return importlib.import_module("definers.application_chat.handlers")

    @classmethod
    def get_contract(cls, name):
        return getattr(cls.get_contracts_module(), name)

    @classmethod
    def get_handler_export(cls, name):
        return getattr(cls.get_handlers_module(), name)


Answerer = ApplicationChatFacade.get_contract("Answerer")
ChatAuditMessageBuilder = ApplicationChatFacade.get_contract(
    "ChatAuditMessageBuilder"
)
ChatHistory = ApplicationChatFacade.get_contract("ChatHistory")
ChatMessage = ApplicationChatFacade.get_contract("ChatMessage")
ChatRequest = ApplicationChatFacade.get_contract("ChatRequest")
ChatRequestNormalizer = ApplicationChatFacade.get_contract(
    "ChatRequestNormalizer"
)
HistoryItem = ApplicationChatFacade.get_contract("HistoryItem")
LanguageDetector = ApplicationChatFacade.get_contract("LanguageDetector")
Logger = ApplicationChatFacade.get_contract("Logger")
Translator = ApplicationChatFacade.get_contract("Translator")
Validator = ApplicationChatFacade.get_contract("Validator")
ChatRequestContext = ApplicationChatFacade.get_handler_export(
    "ChatRequestContext"
)
ChatRequestHandler = ApplicationChatFacade.get_handler_export(
    "ChatRequestHandler"
)
build_chat_audit_message = ApplicationChatFacade.get_handler_export(
    "build_chat_audit_message"
)
create_chat_request = ApplicationChatFacade.get_handler_export(
    "create_chat_request"
)
create_chat_request_handler = ApplicationChatFacade.get_handler_export(
    "create_chat_request_handler"
)
handle_chat_request = ApplicationChatFacade.get_handler_export(
    "handle_chat_request"
)
normalize_chat_request = ApplicationChatFacade.get_handler_export(
    "normalize_chat_request"
)

__all__ = [
    "Answerer",
    "ChatHistory",
    "ChatMessage",
    "ChatAuditMessageBuilder",
    "ChatRequest",
    "ChatRequestContext",
    "ChatRequestHandler",
    "ChatRequestNormalizer",
    "HistoryItem",
    "LanguageDetector",
    "Logger",
    "Translator",
    "Validator",
    "build_chat_audit_message",
    "create_chat_request",
    "create_chat_request_handler",
    "handle_chat_request",
    "normalize_chat_request",
]
