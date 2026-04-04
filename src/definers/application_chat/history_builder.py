from definers.application_chat.contracts import (
    ChatMediaAttachment,
    ChatTextContext,
    HistoryItem,
)


class ChatHistoryBuilder:
    media_instruction = (
        "and please read the media from my new message carefully"
    )

    @classmethod
    def build_history(
        cls,
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
                    {"role": "user", "content": cls.media_instruction}
                )
        return normalized_history, tuple(included_types)
