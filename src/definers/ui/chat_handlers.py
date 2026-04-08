class ChatPresentationService:
    @staticmethod
    def validate_text_input(value):
        from definers.text.validation import TextInputValidator

        return TextInputValidator.default().validate(value)

    @staticmethod
    def build_chat_audit_message(included_types, original_language):
        from definers.chat.handlers import (
            build_chat_audit_message as build_application_chat_audit_message,
        )
        from definers.constants import language_codes

        return build_application_chat_audit_message(
            included_types,
            original_language,
            language_codes,
        )

    @staticmethod
    def create_default_chat_request_handler():
        import definers.text as text
        from definers.chat.handlers import (
            create_chat_request_handler,
        )
        from definers.constants import language_codes
        from definers.system import log
        from definers.text.validation import TextInputValidator

        validator = TextInputValidator.default()
        current_answer = answer
        if current_answer is None:
            from definers.ml import answer as current_answer

        return create_chat_request_handler(
            detect_language=text.language,
            translate_text=text.ai_translate,
            validate_text=validator.validate,
            answer=current_answer,
            log=log,
            language_names=language_codes,
        )

    @staticmethod
    def to_chat_request(message, history):
        from definers.chat.handlers import create_chat_request

        return create_chat_request(message, history)

    @staticmethod
    def get_chat_response(message, history):
        import definers.text as text
        from definers.chat.handlers import handle_chat_request
        from definers.constants import language_codes
        from definers.system import log
        from definers.text.validation import TextInputValidator

        validator = TextInputValidator.default()
        current_answer = answer
        if current_answer is None:
            from definers.ml import answer as current_answer

        return handle_chat_request(
            message,
            history,
            detect_language=text.language,
            translate_text=text.ai_translate,
            validate_text=validator.validate,
            answer=current_answer,
            log=log,
            language_names=language_codes,
        )


validate_text_input = ChatPresentationService.validate_text_input
build_chat_audit_message = ChatPresentationService.build_chat_audit_message
create_default_chat_request_handler = (
    ChatPresentationService.create_default_chat_request_handler
)
to_chat_request = ChatPresentationService.to_chat_request
get_chat_response = ChatPresentationService.get_chat_response
answer = None
