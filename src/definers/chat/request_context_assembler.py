class ChatRequestContextAssembler:
    @staticmethod
    def assemble_request_context(
        normalized_request,
        *,
        detect_language,
        translate_text,
        validate_text,
    ):
        from definers.chat.contracts import (
            ChatRequestContext,
            ChatRequestMetadata,
        )
        from definers.chat.history_builder import ChatHistoryBuilder
        from definers.chat.request_factory import ChatRequestFactory
        from definers.chat.text_context_builder import (
            ChatTextContextBuilder,
        )

        base_history = ChatRequestFactory.copy_history(
            normalized_request.history
        )
        media = ChatRequestFactory.normalize_media(normalized_request.message)
        text_context = ChatTextContextBuilder.normalize_text(
            normalized_request.message,
            detect_language=detect_language,
            translate_text=translate_text,
            validate_text=validate_text,
        )
        normalized_history, included_types = ChatHistoryBuilder.build_history(
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
