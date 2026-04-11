from collections.abc import Sequence

from definers.chat.contracts import (
    ChatHistory,
    ChatMediaAttachment,
    ChatMessage,
    ChatRequest,
    HistoryItem,
)


class ChatRequestFactory:
    @staticmethod
    def coerce_chat_request(
        request: ChatRequest | ChatMessage,
        history: ChatHistory | None,
    ) -> ChatRequest:
        if isinstance(request, ChatRequest):
            return ChatRequestFactory.create_chat_request(
                request.message,
                request.history,
            )
        return ChatRequestFactory.create_chat_request(request, history)

    @staticmethod
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

    @staticmethod
    def copy_history(history: ChatHistory) -> list[HistoryItem]:
        return [dict(item) for item in history]

    @staticmethod
    def normalize_files(message: ChatMessage) -> tuple[str, ...]:
        files = message.get("files", [])
        if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
            return ()
        normalized_files: list[str] = []
        for file_path in files:
            normalized_file_path = str(file_path).strip()
            if normalized_file_path:
                normalized_files.append(normalized_file_path)
        return tuple(normalized_files)

    @classmethod
    def normalize_media(
        cls,
        message: ChatMessage,
    ) -> tuple[ChatMediaAttachment, ...]:
        return tuple(
            ChatMediaAttachment(path=file_path)
            for file_path in cls.normalize_files(message)
        )
