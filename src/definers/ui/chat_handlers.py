def validate_text_input(value):
    from definers.text.validation import TextInputValidator

    return TextInputValidator.default().validate(value)


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


def to_chat_request(message, history):
    from definers.chat.handlers import create_chat_request

    return create_chat_request(message, history)


def _complete_chat_request(message, history):
    from definers.chat.handlers import (
        ChatRequestAuditLogger,
        ChatRequestFactory,
    )

    request_handler = create_default_chat_request_handler()
    normalized_request = ChatRequestFactory.coerce_chat_request(
        message,
        history,
    )
    context = request_handler.normalize_request(normalized_request)
    ChatRequestAuditLogger.log_request(
        request_handler.log,
        request_handler.build_audit_message,
        context,
    )
    response = request_handler.answer(context.history)
    ChatRequestAuditLogger.log_response(request_handler.log, response)
    return response


def get_chat_response(message, history):
    return _complete_chat_request(message, history)


def get_chat_response_stream(message, history):
    from definers.chat.handlers import (
        ChatRequestAuditLogger,
        ChatRequestFactory,
    )

    request_handler = create_default_chat_request_handler()
    yield "Validating chat request..."
    normalized_request = ChatRequestFactory.coerce_chat_request(
        message,
        history,
    )
    yield "Normalizing chat context..."
    context = request_handler.normalize_request(normalized_request)
    yield "Logging request context..."
    ChatRequestAuditLogger.log_request(
        request_handler.log,
        request_handler.build_audit_message,
        context,
    )
    yield "Running answer runtime..."
    response = request_handler.answer(context.history)
    yield "Finalizing response..."
    ChatRequestAuditLogger.log_response(request_handler.log, response)
    yield response


answer = None
