from definers.application_chat.contracts import (
    ChatMessage,
    ChatTextContext,
    LanguageDetector,
    Translator,
    Validator,
)


class ChatTextContextBuilder:
    @staticmethod
    def normalize_text(
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
